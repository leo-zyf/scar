"""Compose CSR update and solve in one repeated resource and launch plan."""

import argparse
import json
from pathlib import Path
from statistics import median

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import (CostEstimate, ImplementationOption, PlanningOperation,
    PlanningProblem, PlanningResource, problem_document)
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import decode
from scar.optimization.constructive_planner import construct_implementations
from scar.runtime.csr_region_entry import SELECTION as ROW
from scar.runtime.sparse_pipeline_launch import SCHEMA
from scar.runtime.sparse_solve_launch import LAUNCH as SOLVE

SOURCE = 'scar.sparse-pipeline-scope'
VALIDATION = 'scar.sparse-pipeline-entry-validation'
EXECUTION = 'scar.sparse-pipeline-execution'
TEMPLATE = 'scar.sparse-pipeline-loop-template'
MODES = (('KEEP', 'original'), ('SOLVE_ONLY', 'solve_only'), ('JOINT', 'joint'))
FIELDS = ('estimate_ns', 'lower_ns', 'upper_ns')


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _reference(path, schema):
    return artifact_reference(Path(path).resolve(), schema)


def _write(directory, name, value):
    path = directory/name
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return _reference(path, value['schema'])


def prepare(scope_path, row_path, solve_path, out):
    from scar.analysis.sparse_pipeline_scope import check_sparse_pipeline_scope

    source = _reference(scope_path, SOURCE)
    scope = check_sparse_pipeline_scope(source)
    row, solver = _reference(row_path, ROW), _reference(solve_path, SOLVE)
    update, qualified = read_artifact(row), read_artifact(solver)
    solve = read_artifact(qualified['study'])
    entry = scope['entry']['source']
    context = qualified['context']
    _check(entry == update['entry'] == qualified['entry'] == solve['entry']
          and context == solve['launch_context'] and scope['entry']['argv'] == context['argv']
          and scope['entry']['affinity'] == context['cpu_ids']
          and qualified['decision'] == 'LOAD_NATIVE_CANDIDATE' and update['topology']['proof'] == scope['topology_scope']
          and solve['source'] == scope['solve_scope'], 'component evidence does not describe one admitted pipeline')
    runtime = Path(__file__).resolve().parents[1]/'runtime'
    members = [_reference(runtime/name, 'scar.python-source')
               for name in ('sparse_pipeline_launch.py', 'sparse_pipeline_entry.py')]
    count = read_artifact(scope['source_scope'])['observed']['consumer_completions']
    _check(type(count) is int and count > 1, 'exact repeated pipeline invocation count required')
    budgets = dict(row=update['topology']['budget_bytes'], solve=solve['budget_bytes'])
    directory = Path(out).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    initial = dict(schema=SCHEMA, schema_version=1, entry=entry, context=context,
        composition=source, row_selection=row, solve_launch=solver, solve_selection=qualified['study'],
        runtime_members=members, count=count, budgets=budgets, study_joint=True, decision='SOLVE_ONLY')
    return _write(directory, 'selection.json', initial)


def _loop_template(study, cost, cost_ref):
    scope = read_artifact(study['composition'])
    source = study['composition']
    return dict(schema=TEMPLATE, schema_version=1, source=study['composition'], cost=cost_ref,
        count=study['count'], unrolled=False,
        body=dict(operations=[dict(id='row-update', artifact=source, locator='#/regions/row_edit'),
                             dict(id='source-normalization-and-solve', artifact=source, locator='#/regions/sparse_solve')],
            causal_edge=['row-update.normal-exit', 'source-normalization-and-solve.entry'],
            value_edge_sources=[dict(artifact=source, locator='#/value_routes/'+key) for key in scope['value_routes']],
            normalization=dict(artifact=source, locator='#/normalization')),
        backedge=dict(before='source-normalization-and-solve.normal-exit', after='row-update.entry',
            iteration_distance=1, original_control=dict(artifact=source, locator='#/loop'),
            preserves_original_caller_work=True),
        cpu=dict(ids=study['context']['cpu_ids'], execution_lanes=1),
        owned_metadata=dict(component_caps=study['budgets'], observed=cost['resource'],
            same_scope_lifetime='FROM_FIRST_USE_THROUGH_ENTRY_EXIT'),
        timings=dict(artifact=cost_ref, locator='#/timings_by_mode'),
        cost_projection='ADMIT + FIRST_PAIR + (COUNT - 1) * WARM_PAIR + CLOSE_SCOPE',
        loaders_are_nested_in_first_pair=True, other_entry_services_ns=None)


def _plan(study, template_ref, target, cost, cost_ref):
    refs = (dict(artifact=study['composition'], locator='#'), dict(artifact=template_ref, locator='#'))
    measured = (dict(artifact=cost_ref, locator='#/timings_by_mode'),)
    lane, pool = 'calling-cpu-thread', 'pipeline-owned-metadata'
    options = []
    for option, mode in MODES:
        projection = cost['conditional_serial_scope'][mode]
        bounds = tuple(projection[key] for key in FIELDS)
        retained = cost['resource'][mode]['peak_simultaneous_persistent_bytes']
        _check(retained <= sum(study['budgets'].values()), 'observed payload exceeds the allowed metadata budget')
        bounded = CostEstimate(*bounds, 'EMPIRICAL_MODEL', cost_ref['sha256'], measured)
        options.append(ImplementationOption(option, option, refs, {lane:1},
            {pool:retained} if retained else {}, {}, {}, bounded,
            dict(loop_template=template_ref, call_count=study['count'], cpu_ids=study['context']['cpu_ids'])))
    resources = (PlanningResource(lane, 'EXECUTION', 1, refs),
                 PlanningResource(pool, 'MEMORY', sum(study['budgets'].values()), refs))
    op = PlanningOperation('repeated-sparse-pipeline:'+study['composition']['sha256'], refs, (), tuple(options), 'KEEP')
    return PlanningProblem(decode(ProviderArtifact, target), refs, resources, (op,), (),
        assumptions=('COMPACT_SERIAL_LOOP_BODY', 'SIMULTANEOUS_PERSISTENT_PAYLOAD',
                     'SAMPLE_ENVELOPES_NOT_PHYSICAL_BOUNDS', 'OUTSIDE_BODY_AND_NUMERIC_WORKSPACE_UNQUALIFIED'))


def _validation(benchmark, study_ref, study, cost):
    _check(benchmark['schema_version'] == 1 and benchmark['selection'] == study_ref
           and benchmark['context'] == study['context'] and benchmark['clean'] is True,
           'full-process comparisons differ from the admitted launch')
    warm, repeats = benchmark['warmup_pairs'], benchmark['measurement_pairs']
    modes = ('solve_only', 'joint')
    runs = {(r['pair'],r['mode']):r for r in benchmark['runs']}
    _check(type(warm) is int and warm >= 0 and type(repeats) is int and repeats >= 3
          and set(runs) == {(p,m) for p in range(warm+repeats) for m in modes}
          and len(runs) == len(benchmark['runs']), 'matched full-process pairs required')
    _check(len({(r['stdout_sha256'],r['stderr_sha256']) for r in runs.values()}) == 1,
           'composed execution changed the observable result')
    timings = {mode: [] for mode in modes}
    count = study['count']
    for (pair, mode), row in sorted(runs.items()):
        observed = read_artifact(row['launch'])
        joint = mode == 'joint'
        resources = cost['resource'][mode]
        _check(row['launch']['schema'] == EXECUTION
               and observed['launch']['selection_sha256'] == study_ref['sha256'] and observed['exit_code'] == 0
               and observed['entry_path'] == study['entry']['uri'] and observed['argv'] == study['context']['argv']
               and observed['decision'] == ('ROW_AND_SOLVE' if joint else 'SOLVE_ONLY'),
               'validation did not execute the intended pipeline bundle')
        for stage, calls in (('row', count if joint else 0), ('solve', count)):
            stage_report = observed['stages'][stage]
            counters = stage_report['statistics']
            _check(counters['native_calls'] == calls and counters['fallback_calls'] == 0,
                   'validation did not cover every guarded pipeline invocation')
            if calls:
                _check(stage_report['release']['closed'] is True
                       and stage_report['release']['retained_after_close_bytes'] == 0,
                       'validation did not release an active plan')
        solve = observed['stages']['solve']['symbolic']
        _check(solve['builds'] == 1 and solve['hits'] == count-1 and solve['retained_bytes'] ==
               resources['symbolic_retained_bytes'], 'symbolic plan lifetime differs')
        if joint:
            update = observed['stages']['row']['topology']
            _check(update['builds'] == 1 and update['hits'] == count-1 and update['retained_bytes'] ==
                   resources['topology_retained_bytes'], 'row topology lifetime differs')
        _check(type(row['wall_ns']) is int and row['wall_ns'] > 0, 'positive full-process clock required')
        if pair >= warm:
            timings[mode].append(row['wall_ns'])
    return timings


def qualify(study_path, cost_path, validation_path, out):
    from scar.optimization.sparse_pipeline_cost import SCHEMA as COST, check

    study_ref = _reference(study_path, SCHEMA)
    study = read_artifact(study_ref)
    cost_ref = _reference(cost_path, COST)
    cost = check(cost_ref['uri'])
    design = read_artifact(cost['design'])
    _check(cost['composition'] == study['composition'] and cost['context'] == study['context']
           and design['update_selection'] == study['row_selection'] and design['solve_selection'] == study['solve_selection']
           and cost['iteration_template']['count'] == study['count']
           and cost['qualification']['all_values_bitwise'] and cost['qualification']['all_expected_attachments'],
           'paired costs describe another compositional launch')
    validation_ref = _reference(validation_path, VALIDATION)
    observed = _validation(read_artifact(validation_ref), study_ref, study, cost)
    bucket = Path(out).resolve()
    bucket.mkdir(parents=True, exist_ok=True)
    template = _write(bucket, 'loop-template.json', _loop_template(study, cost, cost_ref))
    solver = read_artifact(study['solve_launch'])
    target = read_artifact(solver['problem'])['target']
    problem = _plan(study, template, target, cost, cost_ref)
    plan = construct_implementations(problem)
    problem_ref = _write(bucket, 'problem.json', problem_document(problem))
    plan_ref = _write(bucket, 'plan.json', plan)
    native = observed['joint']
    current = observed['solve_only']
    modeled_floor = cost['conditional_serial_scope']['joint_savings_over_solve_only']['lower_ns']
    selected = any(row['option'] == 'JOINT' for row in plan['selected'])
    enable = selected and modeled_floor > 0 and max(native) < min(current)
    return _write(bucket, 'selection.json', {**study, 'study_joint':False,
        'decision':'JOINT' if enable else 'SOLVE_ONLY', 'template':template, 'problem':problem_ref,
        'plan':plan_ref, 'cost':cost_ref, 'validation':validation_ref, 'study':study_ref,
        'joint_incremental_modeled_lower_ns':modeled_floor,
        'entry_medians_ns':{mode:median(vals) for mode,vals in observed.items()}})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope')
    parser.add_argument('--row-selection')
    parser.add_argument('--solve-launch')
    parser.add_argument('--study')
    parser.add_argument('--cost')
    parser.add_argument('--validation')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if all((args.scope, args.row_selection, args.solve_launch)) and not args.study:
        result = prepare(args.scope, args.row_selection, args.solve_launch, args.out)
    elif all((args.study, args.cost, args.validation)) and not args.scope:
        result = qualify(args.study, args.cost, args.validation, args.out)
    else:
        parser.error('use --scope/--row-selection/--solve-launch or --study/--cost/--validation')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

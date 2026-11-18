"""Project the opaque direct-solver call into structural and numeric work."""

import argparse
import json
import marshal
from pathlib import Path
from statistics import median
import sys

from scar.analysis.sparse_solve_callsite import seal_callsite
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import (CostEstimate, ImplementationChoice, ImplementationOption,
    PlanningActivity, PlanningOperation, PlanningProblem, PlanningResource, PlanningTimingConstraint, problem_document)
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import decode
from scar.optimization.constructive_planner import construct_implementations
from scar.runtime.csr_region_entry import interpreter_context, launch_context
from scar.runtime.sparse_solve_entry import ENTRY_CODE, SELECTION
from scar.runtime.sparse_solve_launch import LAUNCH
from scar.trace.code_snapshot import code_key

SOURCE = 'scar.sparse-solve-scope'
NATIVE = 'scar.superlu-symbolic-study'
DECOMPOSITION = 'scar.sparse-direct-solve-decomposition'
PLAN = 'scar.sparse-direct-solve-plan'
VALIDATION = 'scar.sparse-solve-entry-validation'
FIELDS = ('estimate_ns', 'lower_ns', 'upper_ns')


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, value['schema'])


def _stages(scope):
    """Express what would be shared while preserving the original numeric algorithm."""
    return (
        dict(id='column-order', reads=['csr-structure', 'column-policy', 'transpose-mode'],
             produces=['raw-column-permutation'], lifetime=scope, effect='OWNED_STRUCTURAL_METADATA'),
        dict(id='preorder', reads=['csr-structure', 'raw-column-permutation'],
             produces=['final-column-permutation', 'elimination-tree', 'column-slices'],
             lifetime=scope, effect='OWNED_STRUCTURAL_METADATA'),
        dict(id='numeric-factorization', reads=['csr-values', 'current-column-slices', 'elimination-tree',
             'final-column-permutation', 'pivot-policy'], produces=['row-pivots', 'numeric-factors'],
             lifetime='PER_CALL', effect='PER_CALL_WORKSPACE'),
        dict(id='triangular-solve', reads=['right-hand-side', 'row-pivots', 'numeric-factors',
             'final-column-permutation', 'transpose-mode'], produces=['solution', 'status'],
             lifetime='PER_CALL', effect='RETAIN_ORIGINAL_STATUS_AND_WARNING_PATH'),
    )


def _problem(source, target, decomposition, budget_bytes, cost_ref=None, cost=None, *, paired=None, paired_ref=None):
    reference = (dict(artifact=source, locator='#/source_call'), dict(artifact=decomposition, locator='#'))
    lane, pool = 'calling-cpu-thread', 'owned-symbolic-metadata'
    operation = 'sparse-direct-solve:'+source['sha256']
    count = read_artifact(decomposition)['call_count']
    if paired is not None:
        timings = paired['timings']
        measured_ref = (dict(artifact=paired_ref, locator='#/timings'),)

        def estimate(values):
            return CostEstimate(*values, 'EMPIRICAL_MODEL', paired_ref['sha256'], measured_ref)

        def series(mode):
            return estimate(tuple(timings[mode+'_first'][key] + (count-1)*timings[mode+'_warm'][key]
                                  for key in FIELDS))

        baseline, candidate = series('original'), series('native')
        when = (ImplementationChoice(operation, 'REUSE_SYMBOLIC'),)
        activities = tuple(PlanningActivity(name, estimate(tuple(timings[name][key] for key in FIELDS)),
            {lane:1}, measured_ref, when) for name in ('load_once', 'close_once'))
        constraints = (
            PlanningTimingConstraint('solver-after-native-load', 'load_once', 'END', operation, 'START',
                when=when, source_refs=measured_ref),
            PlanningTimingConstraint('free-symbolic-after-solve', operation, 'END', 'close_once', 'START',
                when=when, source_refs=measured_ref),
        )
        reserved = paired['resource']['symbolic_retained_bytes']
    else:
        candidate = CostEstimate(None, None, None, 'UNKNOWN', source['sha256'], reference,
            ('ORIGINAL_PROVIDER_INTERNAL_PHASES_UNMEASURED', 'SCAR_SYMBOLIC_SCOPE_COST_NOT_QUALIFIED'))
        activities, constraints, reserved = (), (), budget_bytes
    if paired is None and (cost is None or count is None):
        baseline = CostEstimate(None, None, None, 'UNKNOWN', source['sha256'], reference,
            ('MISSING_BASELINE_SCOPE_COST',))
    elif paired is None:
        timing = cost['timings']['direct_csr']
        baseline = CostEstimate(*[count*timing[key] for key in FIELDS],
            'EMPIRICAL_MODEL', cost_ref['sha256'], (dict(artifact=cost_ref, locator='#/timings/direct_csr'),))
    return PlanningProblem(decode(ProviderArtifact, target), reference,
        (PlanningResource(lane, 'EXECUTION', 1, reference), PlanningResource(pool, 'MEMORY', budget_bytes, reference)),
        (PlanningOperation(operation, reference, (), (
            ImplementationOption('KEEP', 'KEEP', reference, {lane:1}, {}, {}, {}, baseline,
                dict(call_count=count, includes_provider_setup_each_call=True)),
            ImplementationOption('REUSE_SYMBOLIC', 'REUSE_SYMBOLIC', reference, {lane:1}, {pool:reserved}, {}, {},
                candidate, dict(call_count=count, format='CSR', numeric_factorization='PER_CALL',
                transpose_mode_preserved=True, full_entry_qualified_here=False)),
        ), 'KEEP'),), (), assumptions=('SERIAL_CPU_SCOPE', 'EXACT_CASE_CALL_COUNT_ONLY',
          'SOURCE_SOLVE_SECTION_ESTIMATE_NOT_COMPLETE_ENTRY', 'NATIVE_HEAP_AND_NUMA_UNQUALIFIED'),
        activities=activities, timing_constraints=constraints)


def prepare(scope_path, out, *, cost_path=None, native_path=None, superlu_source=None, blas=None,
            jobs=4, budget_bytes=8*1024*1024):
    from scar.analysis.sparse_solve_scope import check_sparse_solve_scope

    if (superlu_source is None) != (blas is None) or (native_path is not None and superlu_source is not None):
        raise ValueError('choose --native or both --superlu-source and --blas')
    if type(budget_bytes) is not int or budget_bytes <= 0:
        raise ValueError('positive native symbolic payload limit required')
    source_ref = artifact_reference(Path(scope_path).resolve(), SOURCE)
    scope = check_sparse_solve_scope(source_ref)
    entry = scope['entry']
    call = scope['source_call']
    source = call['source']
    original = read_artifact(source, json_document=False)
    carrier = call['carrier']['library']
    read_artifact(carrier, json_document=False)
    site, code = seal_callsite(original, source['uri'], call['code_key'], call['source_span'],
        call['carrier']['name'], optimize=sys.flags.optimize)
    context = {**launch_context(entry['argv']), 'cpu_ids': entry['affinity']}
    cost_ref = cost = None
    if cost_path is not None:
        from scar.optimization.sparse_solve_cost import SCHEMA, check

        cost_ref = artifact_reference(Path(cost_path).resolve(), SCHEMA)
        cost = check(cost_ref['uri'])
        if cost['source_scope'] != scope['source_scope'] or cost['context'] != context:
            raise ValueError('solver cost belongs to a different source or launch scope')

    directory = Path(out).resolve()
    native = None
    if native_path is not None or superlu_source is not None:
        from scar.codegen.sparse_lu import PROVIDER_VERSION, build_symbolic_solve, validate_symbolic_solve

        elf = read_artifact(scope['elf_evidence'])
        if (elf['library'] != carrier
                or elf['documentation']['matched_tag'] != f'scipy-v{PROVIDER_VERSION}'):
            raise ValueError('original sparse call lacks the supported SuperLU provider version')
        bundle = Path(native_path).resolve() if native_path is not None else directory / 'native'
        if superlu_source is not None:
            build_symbolic_solve(superlu_source, bundle, blas=blas, jobs=jobs, carrier=carrier)
        native = artifact_reference(bundle / 'artifact.json', NATIVE)
        validate_symbolic_solve(native, carrier)
    directory.mkdir(parents=True, exist_ok=True)
    source_scope = read_artifact(scope['source_scope'])
    count = source_scope['observed']['consumer_completions']
    graph = read_artifact(scope['source_graph'])
    build = read_artifact(graph['program'])
    hardware = artifact_reference(build['target']['path'], 'scar.target-hardware-graph')
    if hardware['sha256'] != build['target']['sha256']:
        raise ValueError('source target hardware graph changed')
    decomposition = _write(directory/'decomposition.json', dict(schema=DECOMPOSITION, schema_version=1,
        source=source_ref, original_call=call, observed_graph=scope['source_graph'], call_count=count,
        cpu_ids=entry['affinity'], stages=_stages('ONE_SCOPED_ACTIVATION_WITH_EXACT_STRUCTURE_GUARD'),
        baseline_phase_costs_ns=None, value_changes_keep_factorization=True,
        candidate_uses_owned_structural_only=True, full_entry_timing_qualified=False))
    problem = _problem(source_ref, hardware, decomposition, budget_bytes, cost_ref, cost)
    problem_ref = _write(directory/'problem.json', problem_document(problem))
    plan = _write(directory/'plan.json', construct_implementations(problem))
    result = dict(schema=PLAN, schema_version=1, source=source_ref,
        decomposition=decomposition, problem=problem_ref, plan=plan, cost=cost_ref,
        decision='KEEP', reason='NATIVE_SYMBOLIC_ENTRY_NOT_QUALIFIED')
    if native is not None:
        code_path = directory/'entry-code.bin'
        with code_path.open('xb') as stream:
            marshal.dump(code, stream)
        runtime = Path(__file__).resolve().parents[1]/'runtime'
        members = [artifact_reference(runtime/name, 'scar.python-source')
                   for name in ('sparse_solve_entry.py', 'sparse_lu.py', 'sparse_lu_native.py')]
        result['study'] = _write(directory/'selection.json', dict(schema=SELECTION, schema_version=1,
            entry=entry['source'], source=source_ref, explicit_study_only=True, decision='KEEP',
            callsite={**call, **site}, launch_context=context, native=native, budget_bytes=budget_bytes,
            runtime_members=members, entry_code=dict(artifact=artifact_reference(code_path, ENTRY_CODE),
                interpreter=interpreter_context(), code_key=code_key(code))))
    return _write(directory/'result.json', result)


def _entry_validation(benchmark, study, context, count, symbolic_bytes):
    if (benchmark['schema_version'] != 1 or benchmark['selection'] != study
            or benchmark['clean'] is not True or benchmark['context'] != context):
        raise ValueError('entry measurements belong to another solve study')
    warm, repeat = benchmark['warmup_pairs'], benchmark['measurement_pairs']
    if type(warm) is not int or warm < 0 or type(repeat) is not int or repeat < 3:
        raise ValueError('matched clean-process solve samples required')
    modes = ('original', 'native')
    runs = {(row['pair'], row['mode']):row for row in benchmark['runs']}
    if len(runs) != len(benchmark['runs']) or set(runs) != {(p,m) for p in range(warm+repeat) for m in modes}:
        raise ValueError('entry comparison pairs are incomplete')
    if len({(row['stdout_sha256'], row['stderr_sha256']) for row in runs.values()}) != 1:
        raise ValueError('original and symbolic entry outputs differ')
    durations = {mode: [] for mode in modes}
    entry = read_artifact(study)['entry']['uri']
    for (pair, mode), row in sorted(runs.items()):
        execution = read_artifact(row['launch'])
        native = mode == 'native'
        stats = execution['statistics']
        if (execution['entry_path'] != entry or execution['argv'] != context['argv'] or execution['exit_code'] != 0
                or execution['selection_sha256'] != study['sha256'] or stats['fallback_calls'] != 0
                or stats['native_calls'] != (count if native else 0)
                or type(row['wall_ns']) is not int or row['wall_ns'] <= 0):
            raise ValueError('entry measurements did not cover the verified solve scope')
        if native:
            kernel = execution['symbolic']
            if (execution['activation']['attempts'] != 1 or kernel['builds'] != 1 or kernel['hits'] != count-1
                    or kernel['retained_bytes'] != symbolic_bytes or kernel['initial_reference_ns'] <= 0
                    or any(kernel[key] for key in ('provider_mismatches', 'numeric_failures', 'structure_misses'))):
                raise ValueError('same symbolic plan was not retained for the whole solve scope')
        elif execution['activation'] is not None:
            raise ValueError('baseline unexpectedly loaded a native solver')
        if pair >= warm:
            durations[mode].append(row['wall_ns'])
    return durations


def qualify(plan_path, cost_path, validation_path, out):
    from scar.optimization.sparse_symbolic_cost import SCHEMA, check

    plan_ref = artifact_reference(Path(plan_path).resolve(), PLAN)
    previous = read_artifact(plan_ref)
    study_ref = previous['study']
    study = read_artifact(study_ref)
    scope = read_artifact(previous['source'])
    decomposition = read_artifact(previous['decomposition'])
    budget = study['budget_bytes']
    count = decomposition['call_count']
    paired_ref = artifact_reference(Path(cost_path).resolve(), SCHEMA)
    paired = check(paired_ref['uri'])
    symbolic_bytes = paired['resource']['symbolic_retained_bytes']
    if (paired['solve_scope'] != previous['source'] or paired['source_scope'] != scope['source_scope']
            or paired['native'] != study['native'] or paired['runtime_adapter'] not in study['runtime_members']
            or paired['context'] != study['launch_context'] or paired['budget_bytes'] != budget
            or not 0 < symbolic_bytes <= budget or paired['qualification']['all_bitwise'] is not True):
        raise ValueError('paired solver measurements differ from the C++ study or source scope')
    validation_ref = artifact_reference(Path(validation_path).resolve(), VALIDATION)
    durations = _entry_validation(read_artifact(validation_ref), study_ref, paired['context'], count, symbolic_bytes)
    original, native = (durations[key] for key in ('original', 'native'))
    target = read_artifact(previous['problem'])['target']
    problem = _problem(previous['source'], target, previous['decomposition'], budget, paired=paired, paired_ref=paired_ref)
    schedule = construct_implementations(problem)
    timing = paired['timings']
    original_floor = timing['original_first']['lower_ns'] + (count-1)*timing['original_warm']['lower_ns']
    native_ceiling = (sum(timing[key]['upper_ns'] for key in ('load_once', 'native_first', 'close_once'))
                      + (count-1)*timing['native_warm']['upper_ns'])
    eligible = (native_ceiling < original_floor and max(native) < min(original)
                and any(row['option'] == 'REUSE_SYMBOLIC' for row in schedule['selected']))
    directory = Path(out).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    problem_ref = _write(directory/'problem.json', problem_document(problem))
    schedule_ref = _write(directory/'plan.json', schedule)
    return _write(directory/'launch.json', dict(schema=LAUNCH, schema_version=1, basis=plan_ref, study=study_ref,
        entry=study['entry'], context=paired['context'], problem=problem_ref, plan=schedule_ref, cost=paired_ref,
        validation=validation_ref, median_ns=dict(original=median(original), native=median(native)),
        modeled_intervals_ns=dict(original_lower=original_floor, native_upper=native_ceiling),
        decision='LOAD_NATIVE_CANDIDATE' if eligible else 'KEEP'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope')
    parser.add_argument('--cost')
    implementation = parser.add_mutually_exclusive_group()
    implementation.add_argument('--native')
    implementation.add_argument('--superlu-source')
    parser.add_argument('--blas')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--out', required=True)
    parser.add_argument('--budget-bytes', type=int, default=8*1024*1024)
    parser.add_argument('--qualify-plan')
    parser.add_argument('--paired-cost')
    parser.add_argument('--validation')
    args = parser.parse_args(argv)
    if bool(args.superlu_source) != bool(args.blas):
        parser.error('--superlu-source and --blas must be supplied together')
    if (args.native or args.superlu_source) and not args.scope:
        parser.error('native implementation options require --scope')
    if args.qualify_plan and args.paired_cost and args.validation and not args.scope:
        output = qualify(args.qualify_plan, args.paired_cost, args.validation, args.out)
    elif args.scope and not any((args.qualify_plan, args.paired_cost, args.validation)):
        output = prepare(args.scope, args.out, cost_path=args.cost, native_path=args.native,
            superlu_source=args.superlu_source, blas=args.blas, jobs=args.jobs, budget_bytes=args.budget_bytes)
    else:
        parser.error('supply --scope or all of --qualify-plan, --paired-cost, --validation')
    print(json.dumps(output))


if __name__ == '__main__':
    main()

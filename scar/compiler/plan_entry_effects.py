"""Replay source-bound external calls around a CPU region into a planning envelope."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from scar.compiler.bind_time_resources import SCHEMA as RESOURCES, read_resource_bindings
from scar.compiler.time_schedule import SCHEMA as TIME
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import (
    CostEstimate, ImplementationOption, PlanningOperation, PlanningProblem,
    PlanningTimingConstraint, ResourceObligation, SCHEMA as PROBLEM, digest,
    problem_document, problem_from_dict, validate_problem,
)
from scar.ir.temporal_resources import projection_from_dict
from scar.optimization.constructive_planner import construct_implementations
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.entry-effect-planning-preparation'
_PREFIX = 'WRITE_SCOPE_EXIT_BEFORE_FIRST_REGION_ENTRY'
_SUFFIX = 'LAST_REGION_EXIT_BEFORE_WRITE_SCOPE_ENTRY'


def _ref(artifact, locator='#'):
    return dict(artifact=artifact, locator=locator)


def _fixed_operation(identity, predecessors, refs, *, kind, parameters, missing, obligations):
    cost = CostEstimate(None, None, None, 'UNKNOWN', identity, refs, (missing,))
    option = ImplementationOption(
        identity + ':original', kind, refs, {}, {}, {}, {}, cost,
        dict(parameters, lowering_authorized=False),
        tuple(ResourceObligation(field, reason, refs) for field, reason in obligations),
    )
    return PlanningOperation(identity, refs, tuple(predecessors), (option,), option.id)


def _effect_orders(region, order, time_ref):
    effects, constraints, frontiers = [], [], []
    groups = defaultdict(list)
    for index, effect in enumerate(order['related_writes']):
        relation = effect['order']
        if relation not in (_PREFIX, _SUFFIX) or effect['service_duration_ns'] is not None:
            raise ValueError('only source-bound prefix or suffix calls with unknown service admitted')
        identity = 'original-call:' + semantic_sha256(dict(
            port=effect['original_source_port'], attempt=effect['call_attempt'],
        ))
        reference = _ref(time_ref, f'#/observed_effects/order/related_writes/{index}')
        row = (identity, reference, effect)
        effects.append(row)
        groups[relation].append(row)
        constraints.append((identity, region, (reference,)) if relation == _PREFIX
                           else (region, identity, (reference,)))

    # Disjoint original calls have a strict order within a phase. A nested
    # interval needs its own hierarchy and cannot be flattened into peer tasks.
    for group in groups.values():
        group.sort(key=lambda row: row[2]['frame_entry']['index'])
        for before, after in zip(group, group[1:]):
            if before[2]['frame_exit']['index'] < after[2]['frame_entry']['index']:
                constraints.append((before[0], after[0], (before[1], after[1])))
            else:
                frontiers.append(dict(reason='OVERLAPPING_ORIGINAL_EFFECT_INTERVALS',
                                      source_refs=[before[1], after[1]]))
    return effects, constraints, frontiers


def _loop_interval(reference, time):
    from scar.analysis.source_ordered_call_content import SCHEMA as LOOP, check
    from scar.analysis.stateful_storage_accesses import SCHEMA as STORAGE

    if reference['schema'] != LOOP:
        raise ValueError('original ordered loop-content projection required')
    loop = check(reference['uri'])
    if loop != read_artifact(reference) or 'observed_interval' not in loop:
        raise ValueError('original loop-content observation changed or is absent')
    interval = loop['observed_interval']
    scope_report = read_artifact(time['observed_effects']['resource_scopes'])
    storage = read_artifact(loop['storage'])
    recipe = read_artifact(time['source'])
    region = read_artifact(recipe['artifacts']['region'])
    if (interval['trace_directory'] != str(Path(scope_report['capture']) / 'trace')
            or loop['storage']['schema'] != STORAGE or storage['catalog']['uri'] != str(Path(scope_report['catalog']) / 'result.json')
            or read_artifact(storage['boundary']) != region):
        raise ValueError('loop-content, external effects, and region must have the same original source and capture')
    result = dict(covered_allocation_interval=interval['allocation_lifetime']['covered_observed_interval'],
        original_call_windows=len(interval['actual_calls']['windows']),
        unresolved_loop_conditions=loop['remaining'])
    if 'control' in interval:
        result['observed_backedges'] = interval['control']['exact_source_backedges']
    return result


def _derive(resource_ref, loop_ref=None, benchmark_ref=None):
    resources = read_resource_bindings(resource_ref['uri'])
    if resources != read_artifact(resource_ref):
        raise ValueError('original resource preparation changed during replay')
    time_ref = resources['time_schedule']
    time = read_artifact(time_ref)
    if (time_ref['schema'] != TIME or time['source_kind'] != 'stateful_recipe'
            or 'observed_effects' not in time):
        raise ValueError('stateful semantic time with source-bound observed effects required')
    projection = projection_from_dict(read_artifact(resources['projection']))
    order = time['observed_effects']['order']
    if order['order_authority'] != 'OBSERVED_NORMAL_OCCURRENCES_ONLY':
        raise ValueError('original observed-order scope required')
    schedule_ref = time['schedule']
    schedule = read_artifact(schedule_ref)  # Fully validated by read_resource_bindings.
    region_id = 'original-region-window:' + semantic_sha256(dict(
        region=schedule['region_id'], source=order['region_source'],
    ))
    effects, boundaries, frontiers = _effect_orders(region_id, order, time_ref)
    if order['open_write_scopes']:
        frontiers.append(dict(reason='WRITE_SCOPE_POSITION_OPEN',
                              source_refs=[_ref(time_ref, '#/observed_effects/order/open_write_scopes')]))
    if not boundaries:
        frontiers.append(dict(reason='NO_RELATED_WRITE_SCOPE',
                              source_refs=[_ref(time_ref, '#/observed_effects/order/related_writes')]))

    predecessors = defaultdict(list)
    constraints = []
    for before, after, refs in boundaries:
        predecessors[after].append(before)
        identity = 'original-order:' + semantic_sha256([before, after])
        constraints.append(PlanningTimingConstraint(
            identity, before, 'END', after, 'START', source_refs=refs,
        ))

    ordered_refs = (_ref(time_ref, '#/observed_effects/order/region_source'),
                    _ref(schedule_ref), _ref(resources['projection']))
    loop_context = None
    if loop_ref is not None:
        loop_context = _loop_interval(loop_ref, time)
        ordered_refs = (*ordered_refs, _ref(loop_ref, '#/observed_interval'))
    region = _fixed_operation(
        region_id, predecessors[region_id], ordered_refs,
        kind='PRESERVE_ORIGINAL_CPU_REGION_WINDOW',
        parameters=dict(
            original_preparation=time_ref, child_schedule=schedule_ref,
            child_region=schedule['region_id'], observed_regions=order['region_occurrences'],
            child_array_budget=dict(bytes=projection.budget.bytes, scope=projection.budget.scope),
            array_task_graph_copied=False, occurrences_expanded=False,
        ),
        missing='WHOLE_ORIGINAL_REGION_WINDOW_SERVICE_NOT_CALIBRATED',
        obligations=(
            ('region_window_cpu_demand', 'The retained child occupancies do not qualify total caller-window CPU demand.'),
            ('region_window_residency', 'The retained array payload and its scoped budget do not bound caller-window residency.'),
        ),
    )
    operations = [region]
    for identity, reference, row in effects:
        operations.append(_fixed_operation(
            identity, predecessors[identity], (reference,),
            kind='PRESERVE_ORIGINAL_EXTERNAL_CALL',
            parameters=dict(original_source_port=row['original_source_port'],
                            observed_side=row['order'], enclosed_positive_writes=len(row['writes'])),
            missing='WHOLE_EXTERNAL_CALL_SERVICE_NOT_CALIBRATED',
            obligations=(('external_service_demand',
                          'Enclosed writes do not qualify caller CPU, storage bandwidth or memory demand.'),),
        ))
    problem = validate_problem(PlanningProblem(
        projection.target, (_ref(resource_ref), _ref(time_ref)), (), tuple(operations), (),
        assumptions=(
            'OBSERVED_NORMAL_OCCURRENCE_ORDER_ONLY',
            'ORIGINAL_CALLS_STAY_OUTSIDE_THE_CHILD_ARRAY_SCHEDULE',
            'OPAQUE_CALLS_AND_UNSEEN_PATHS_DO_NOT_GRANT_MOTION_PERMISSION',
            'WHOLE_SERVICE_UNKNOWN_NO_SYSCALL_TIME_SUBSTITUTION',
        ),
        timing_constraints=tuple(constraints),
    ))
    plan = construct_implementations(problem, baseline_only=True)
    read_artifact(resource_ref)
    replay = dict(
        region_operation=region_id, external_operations=[row[0] for row in effects],
        observed_region_occurrences=order['region_occurrences'],
        boundary_constraints=len(constraints), order_frontiers=frontiers,
        search_status=plan['status'], selected=plan['selected'], makespan_ns=plan['makespan_ns'],
        default_backend=plan['default_backend'], eligibility_frontiers=plan['frontiers'],
        analysis_scope='OBSERVED_ORIGINAL_CPU_ENTRY_ENVELOPE', deployment_authorized=False,
    )
    if loop_context is not None:
        replay['observed_loop_interval'] = loop_context
    if benchmark_ref is not None:
        from scar.optimization.stateful_entry_benchmark import replay_entry_benchmark
        replay['paired_benchmark'] = replay_entry_benchmark(benchmark_ref, time['source'])
    return problem, replay


def review_entry_order(problem, operation_order, *, order_frontiers):
    """Check a proposed serial order against the original fixed-call boundary."""
    operations = {operation.id for operation in problem.operations}
    position = {name: ordinal for ordinal, name in enumerate(operation_order)}
    if len(position) != len(operation_order) or set(position) != operations:
        return dict(status='BLOCKED', reason='FIXED_OPERATIONS_MISSING_REPEATED_OR_UNKNOWN',
                    violated_constraints=[], deployment_authorized=False)
    violated = [edge.id for edge in problem.timing_constraints
                if position[edge.before] >= position[edge.after]]
    status = ('BLOCKED' if violated else 'UNRESOLVED' if order_frontiers
              else 'RETAINS_OBSERVED_BOUNDARIES')
    return dict(status=status, violated_constraints=violated, deployment_authorized=False)


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, value['schema'])


def _report(resources, problem_ref, problem, replay, loop_ref=None, benchmark_ref=None):
    report = dict(schema=SCHEMA, schema_version=1, resources=resources, problem=problem_ref,
                problem_sha256=digest(problem), replay=replay, native_compiled=False,
                target_program_executed=False, performance_measured=False,
                deployment_authorized=False)
    if loop_ref is not None:
        report['loop_content'] = loop_ref
    if benchmark_ref is not None:
        report['paired_benchmark'] = benchmark_ref
    return report


def prepare_entry_effect_plan(resources, out, *, loop_content=None, benchmark=None):
    reference = artifact_reference(Path(resources).resolve(), RESOURCES)
    loop_ref = None
    if loop_content is not None:
        from scar.analysis.source_ordered_call_content import SCHEMA as LOOP
        loop_ref = artifact_reference(Path(loop_content).resolve(), LOOP)
    benchmark_ref = None
    if benchmark is not None:
        from scar.optimization.stateful_entry_benchmark import SCHEMA as BENCHMARK
        benchmark_ref = artifact_reference(Path(benchmark).resolve(), BENCHMARK)
    problem, replay = _derive(reference, loop_ref, benchmark_ref)
    output = Path(out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    problem_ref = _write(output / 'problem.json', problem_document(problem))
    report = _report(reference, problem_ref, problem, replay, loop_ref, benchmark_ref)
    _write(output / 'result.json', report)
    return report


def read_entry_effect_plan(path):
    reference = artifact_reference(Path(path).resolve(), SCHEMA)
    report = read_artifact(reference)
    if type(report.get('schema_version')) is not int or report['schema_version'] != 1:
        raise ValueError('versioned entry-effect planning receipt required')
    loop_ref = report.get('loop_content')
    benchmark_ref = report.get('paired_benchmark')
    problem, replay = _derive(report['resources'], loop_ref, benchmark_ref)
    retained_ref = report['problem']
    expected_ref = artifact_reference(Path(reference['uri']).parent / 'problem.json', PROBLEM)
    if retained_ref != expected_ref or problem_from_dict(read_artifact(retained_ref)) != problem:
        raise ValueError('saved planning graph differs from replayed original call ordering')
    if report != _report(report['resources'], retained_ref, problem, replay, loop_ref, benchmark_ref):
        raise ValueError('saved entry-effect receipt differs from original evidence and planner replay')
    read_artifact(reference)
    return report, problem


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--resources')
    source.add_argument('--check')
    parser.add_argument('--out')
    parser.add_argument('--loop-content', help='replay allocation and call-window evidence from the same capture')
    parser.add_argument('--paired-benchmark', help='replay a matched comparison for validation without rebasing costs')
    parser.add_argument('--proposed-order', nargs='+', metavar='OPERATION')
    args = parser.parse_args(argv)
    if args.check:
        if args.out or args.loop_content or args.paired_benchmark:
            parser.error('--check consumes only the retained receipt')
        report, problem = read_entry_effect_plan(args.check)
    else:
        if not args.out or args.proposed_order:
            parser.error('prepare requires --out and proposed-order review requires --check')
        report = prepare_entry_effect_plan(args.resources, args.out, loop_content=args.loop_content,
                                           benchmark=args.paired_benchmark)
    result = {key: report['replay'][key] for key in (
        'region_operation', 'external_operations', 'boundary_constraints',
        'search_status', 'makespan_ns', 'default_backend', 'deployment_authorized',
    )}
    if args.proposed_order:
        result['proposal'] = review_entry_order(
            problem, args.proposed_order, order_frontiers=report['replay']['order_frontiers'],
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()

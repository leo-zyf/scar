"""L3 finite layout-domain makespan search over replayed L2 publications.

No source, mathematical backend, execution certificate or native host is
generated. Unknown cases stay in the domain and prevent a closed selection.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.pipeline-temporal-domain'
INPUTS_SCHEMA = 'scar.pipeline-temporal-domain-inputs'
RESOURCE_EVENT_DOMAIN = 'scar.pipeline-resource-event-domain'
RESOURCE_EVENT_INPUTS = 'scar.pipeline-resource-event-domain-inputs'
RESOURCE_EVENT_QUERIES = 'scar.pipeline-resource-event-cost-queries'


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, value['schema'])


def _reference(value):
    from .project_pipeline_temporal import SCHEMA as PREPARATION
    if isinstance(value, (str, Path)):
        path = Path(value).resolve()
        value = artifact_reference(path/'result.json' if path.is_dir() else path, PREPARATION)
    return deepcopy(value)


def _common_context(inputs, projection):
    """Only declared layout/placement variables may differ between cases."""
    context = deepcopy(projection['cost_context'])
    for key in ('memory_plan_sha256', 'ready_plan_sha256', 'task_graph_sha256', 'binding',
                'binding_sha256', 'scheduled_slot_capacities', 'tick_seconds'):
        context.pop(key)
    context['original_evidence'].pop('input_graph')
    context['original_evidence'].pop('hardware_bindings_sha256')
    proof = read_artifact(inputs['proof'])
    return dict(program=inputs['program'], source_proof_semantic_sha256=semantic_sha256(proof),
        source_pipeline_sha256=projection['pipeline_sha256'], fixed_context=context)


def _tick(value):
    if value is None:
        return None
    if (type(value) is not dict or set(value) != {'numerator', 'denominator'}
            or any(type(part) is not int or part <= 0 for part in value.values())):
        raise ValueError('exact positive rational tick required')
    result = Fraction(value['numerator'], value['denominator'])
    return dict(numerator=result.numerator, denominator=result.denominator)


def _decision(cases):
    """Pure aggregation; case admission and source replay precede this step."""
    pending = [row['id'] for row in cases if not row['search_complete']]
    feasible = [row for row in cases if row['makespan_ticks'] is not None]
    best = min(feasible, key=lambda row: (row['makespan_ticks'], row['id'])) if feasible else None
    tied = sorted(row['id'] for row in feasible
                  if best is not None and row['makespan_ticks'] == best['makespan_ticks'])
    closed = not pending
    lower = None
    bounds = [row['lower_bound_ticks'] for row in cases if not row['proved_infeasible']]
    if bounds and all(type(bound) is int and bound >= 0 for bound in bounds):
        lower = min(bounds)
    gap = best['makespan_ticks'] - lower if best is not None and lower is not None else None
    return dict(status='UNRESOLVED' if not closed else
        'OPTIMAL_WITHIN_EXPLICIT_DECLARED_DOMAIN' if best is not None else 'INFEASIBLE_WITHIN_EXPLICIT_DECLARED_DOMAIN',
        selection_decision='NO_OP' if not closed or best is None else 'MODEL_SELECTED_EXPLICIT_CASE_ONLY',
        selected_case_id=best['id'] if closed and best else None,
        provisional_best_case_id=best['id'] if best else None,
        provisional_best_makespan_ticks=best['makespan_ticks'] if best else None,
        tied_best_case_ids=tied, tie_policy='MINIMUM_MAKESPAN_THEN_LEXICOGRAPHIC_CASE_ID',
        unresolved_case_ids=pending, all_cases_search_complete=closed,
        lower_bound_ticks=lower, gap_ticks=gap,
        model_optimum_proven=closed and best is not None,
        unique_best_case_proven=closed and len(tied) == 1,
        scope='ONLY_THE_EXPLICIT_FINITE_LAYOUT_AND_FIXED_PLACEMENT_MODELS')


def _case(reference, search_budget):
    from .project_pipeline_temporal import SCHEMA as PREPARATION, open_temporal_inputs
    from scar.optimization.pipeline_temporal_model import (
        verify_pipeline_temporal_projection, plan_pipeline_temporal_model,
    )
    preparation = read_artifact(reference)
    if (preparation.get('schema') != PREPARATION or type(preparation.get('schema_version')) is not int
            or preparation['schema_version'] != 1 or preparation.get('solver_invoked') is not False):
        raise ValueError('published source-only L2 preparation required')
    inputs = read_artifact(preparation['inputs'])
    projection = read_artifact(preparation['projection'])
    with open_temporal_inputs(preparation['inputs']) as (pipeline, memory, ready, target):
        verify_pipeline_temporal_projection(pipeline, memory, ready, target, projection)
        plan = plan_pipeline_temporal_model(pipeline, memory, ready, target,
            cost_model=projection['cost_model'], search_budget=search_budget)
        if semantic_sha256(plan['projection_report']) != semantic_sha256(projection):
            raise ValueError('L3 cost projection differs from retained independently replayed L2')
    read_artifact(reference); read_artifact(preparation['inputs']); read_artifact(preparation['projection'])
    solver = plan['solver_report']
    solution = solver['solution']
    if solution is not None:
        ids = [row['task'] for row in solution['tasks']]
        if len(ids) != len(set(ids)) or set(ids) != set(projection['task_kinds']):
            raise ValueError('solver solution does not cover each original v2 task exactly once')
    optimal = (solver['status'] == 'OPTIMAL_WITHIN_MODEL'
        and solver['optimality'] == 'MODEL_OPTIMUM_ESTABLISHED'
        and solver.get('search_budget_exhausted') is False
        and solver.get('source_schedule_executable') is True and not solver.get('execution_unresolved')
        and not solver['unresolved'] and solution is not None and solver['gap_ticks'] == 0)
    infeasible = (solver['status'] == 'INFEASIBLE_WITHIN_MODEL'
        and solver['optimality'] == 'INFEASIBILITY_ESTABLISHED'
        and solver.get('search_budget_exhausted', False) is False
        and solver.get('source_schedule_executable') is True and not solver.get('execution_unresolved')
        and not solver['unresolved'] and solution is None)
    identifier = 'case:' + semantic_sha256(reference)
    row = dict(id=identifier, preparation=reference, inputs=preparation['inputs'],
        projection=preparation['projection'], original_proof=inputs['proof'],
        memory=inputs['memory'], ready=inputs['ready'], cost_model=projection['cost_model'],
        cost_authority=plan['cost_authority'], tick_seconds=_tick(projection['cost_context']['tick_seconds']),
        solver_status=solver['status'], optimality=solver['optimality'],
        search_complete=optimal or infeasible, proved_infeasible=infeasible,
        search_budget_exhausted=solver.get('search_budget_exhausted', False),
        makespan_ticks=solution['makespan_ticks'] if solution else None,
        lower_bound_ticks=solver['lower_bound_ticks'], gap_ticks=solver['gap_ticks'],
        states_explored=solver['states_explored'], unresolved=deepcopy(solver['unresolved']),
        runtime_order_executed=False, deployment_authorized=False)
    return row, plan, _common_context(inputs, projection)


def _derive(projections, search_budget):
    if type(search_budget) is not int or search_budget < 1:
        raise ValueError('positive per-case solver search budget required')
    if isinstance(projections, (str, Path, dict)):
        projections = [projections]
    references = [_reference(value) for value in projections]
    if not references:
        raise ValueError('nonempty explicit L2 case domain required')
    cases, plans, common, known_tick = [], {}, None, None
    seen = set()
    for reference in references:
        if semantic_sha256(reference) in seen:
            raise ValueError('duplicate L2 preparation in the explicit domain')
        seen.add(semantic_sha256(reference))
        row, plan, context = _case(reference, search_budget)
        if common is None:
            common = context
        elif semantic_sha256(common) != semantic_sha256(context):
            raise ValueError('cases differ in program/source proof/pipeline/target/shape/contract/budget/context')
        tick = row['tick_seconds']
        if tick is not None:
            if known_tick is None:
                known_tick = tick
            elif tick != known_tick:
                raise ValueError('known case costs do not use the same exact rational tick')
        cases.append(row); plans[row['id']] = plan
    # A case without a tick necessarily has unknown durations and remains in
    # this domain. Its bound is never recast as a known zero or silently omitted.
    decision = _decision(cases)
    return dict(schema=SCHEMA, schema_version=1, layer='L3_FINITE_MODEL_MAKESPAN_SEARCH',
        common_contract=common, tick_seconds=known_tick, cases=cases, decision=decision,
        search_budget_per_case=search_budget,
        states_explored=sum(row['states_explored'] for row in cases),
        case_count=len(cases), retained_case_count=len(references),
        all_options_retained=True, no_unknown_cost_imputed=True,
        solver_invoked=True, native_compiled=False, certificate_generated=False,
        runtime_order_executed=False, physical_optimality_proven=False,
        deployment_authorized=False, target_program_executed=False, performance_measured=False), plans


def solve_pipeline_temporal(projections, out, *, search_budget=100000):
    """Replay each L2 case and compare only its explicit, common-unit model."""
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('fresh L3 output directory required')
    if isinstance(projections, (str, Path, dict)):
        projections = [projections]
    declaration = dict(schema=INPUTS_SCHEMA, schema_version=1,
        projections=[_reference(value) for value in projections], search_budget_per_case=search_budget,
        scope='EXACT_EXPLICIT_FINITE_CASE_LIST_DECLARED_BEFORE_SEARCH')
    out.mkdir(parents=True)
    declared = _write(out/'domain-inputs.json', declaration)
    report, plans = _derive(declaration['projections'], search_budget)
    read_artifact(declared)
    report['declared_domain'] = declared
    for index, row in enumerate(report['cases']):
        directory = out/'cases'/str(index); directory.mkdir(parents=True)
        row['plan'] = _write(directory/'plan.json', plans[row['id']])
    _write(out/'result.json', report)
    return report


def verify_pipeline_temporal_domain(report):
    """Reopen original L2/source evidence and rerun every case, without writes."""
    if isinstance(report, (str, Path)):
        path = Path(report).resolve()
        if path.is_dir(): path = path/'result.json'
        value = json.loads(path.read_bytes())
    else:
        value = deepcopy(report)
    if value.get('schema') != SCHEMA or type(value.get('schema_version')) is not int or value['schema_version'] != 1:
        raise ValueError('versioned L3 domain required')
    declaration = read_artifact(value['declared_domain'])
    if (declaration.get('schema') != INPUTS_SCHEMA or type(declaration.get('schema_version')) is not int
            or declaration['schema_version'] != 1):
        raise ValueError('original explicit finite input-domain publication required')
    expected, plans = _derive(declaration['projections'], declaration['search_budget_per_case'])
    expected['declared_domain'] = value['declared_domain']
    for actual, row in zip(value['cases'], expected['cases'], strict=True):
        stored = read_artifact(actual['plan'])
        if semantic_sha256(stored) != semantic_sha256(plans[row['id']]):
            raise ValueError('retained case plan differs from complete independent source/model/solver replay')
        row['plan'] = actual['plan']
    if semantic_sha256(expected) != semantic_sha256(value):
        raise ValueError('finite domain differs from retained cases, contexts, costs or selection replay')
    read_artifact(value['declared_domain'])
    return value


def _file_reference(value, schema, member):
    if isinstance(value, (str, Path)):
        path = Path(value).resolve()
        return artifact_reference(path/member if path.is_dir() else path, schema)
    if type(value) is not dict or value.get('schema') != schema:
        raise ValueError(f'original {schema} artifact reference required')
    return deepcopy(value)


def _domain_case_refs(row, memory, ready, inputs_ref, raw, arena_refs, *, stored=None, output=None):
    from .project_pipeline_temporal import INPUTS

    if row['implementation'] == 'ORIGINAL_PYTHON':
        return dict(inputs=inputs_ref, memory=None, ready=None)
    arena = row['arena_policy']
    memory_ref = arena_refs.get(arena)
    if stored is None:
        folder = output/'cases'/row['id'].replace(':', '-')
        folder.mkdir(parents=True)
        if memory_ref is None:
            memory_file = output/'arenas'/(arena+'.json')
            memory_file.parent.mkdir(exist_ok=True)
            memory_ref = _write(memory_file, memory)
        ready_ref = None if ready is None else _write(folder/'ready.json', ready)
    else:
        ready_ref = stored['ready']
        if memory_ref is not None and stored['memory'] != memory_ref:
            raise ValueError('one original artifact must be shared by all cases of this arena policy')
        if memory_ref is None:
            memory_ref = stored['memory']
        if arena not in arena_refs and semantic_sha256(read_artifact(memory_ref)) != semantic_sha256(memory):
            raise ValueError('retained arena differs from replayed domain case')
        if (ready is None) != (ready_ref is None) or (ready_ref is not None
                and semantic_sha256(read_artifact(ready_ref)) != semantic_sha256(ready)):
            raise ValueError('retained ready plan differs from replayed domain case')
    arena_refs[arena] = memory_ref
    case_inputs = None
    if ready_ref is not None:
        inputs = dict(raw, memory=memory_ref, ready=ready_ref)
        if stored is None:
            case_inputs = _write(folder/'inputs.json', inputs)
        else:
            case_inputs = stored['inputs']
            case_inputs = _file_reference(case_inputs, INPUTS, 'inputs.json')
            if read_artifact(case_inputs) != inputs:
                raise ValueError('case inputs do not retain the original proof and their exact arena/ready references')
    return dict(inputs=case_inputs, memory=memory_ref, ready=ready_ref)


def _resource_event_report(declaration, *, output=None, stored=None):
    from .project_pipeline_temporal import open_temporal_inputs
    from scar.analysis.replay_transaction import replay_transaction
    from scar.optimization.pipeline_implementation import build_pipeline_problem
    from scar.optimization.pipeline_resource_event_domain import domain_cases, cost_context, case_cost, decide

    budget = declaration['search_budget_per_arena']
    if type(budget) is not int or budget < 1:
        raise ValueError('positive finite arena search budget required')
    inputs_ref, profile = declaration['source_inputs'], declaration['cost_profile']
    if profile is not None:
        read_artifact(profile)
    raw = read_artifact(inputs_ref)
    source_refs = tuple(dict(role=role, artifact=raw[field]) for role, field in (
        ('ORIGINAL_F4_PROGRAM', 'program'), ('ORIGINAL_F4_CALLS_AND_EFFECTS', 'proof'),
        ('ORIGINAL_VALUE_VERSIONS', 'source_values')))
    with replay_transaction(), open_temporal_inputs(inputs_ref) as (pipeline, source_memory, source_ready, target):
        context = cost_context(pipeline, source_memory, source_ready, raw)
        derived = list(domain_cases(pipeline, source_memory, source_ready, target, raw['source_values'],
            search_budget=budget))
        if stored is not None and [r['id'] for r in stored] != [case[0]['id'] for case in derived]:
            raise ValueError('retained resource-event domain changed its complete ordered case list')
        cases, leaf_problems, arena_refs = [], {}, {}
        missing_costs, cost_queries = {}, {}
        for index, (row, memory, ready) in enumerate(derived):
            case_refs = _domain_case_refs(row, memory, ready, inputs_ref, raw, arena_refs, output=output,
                                          stored=None if stored is None else stored[index])
            if ready is not None:
                # Every allowed CPU pair starts with the singleton CPU. Share
                # the source-qualified implementation contexts within an arena.
                key = row['arena_policy']
                if key not in leaf_problems:
                    cpus = next((candidate['cpu_ids'] for candidate, _, plan in reversed(derived)
                        if plan is not None and candidate['arena_policy'] == key), row['cpu_ids'])
                    leaf_problems[key] = build_pipeline_problem(pipeline, memory, target, cpu_ids=cpus,
                        source_refs=source_refs)
            row.update(case_refs, original_proof=raw['proof'], source_values=raw['source_values'])
            row['cost'] = case_cost(pipeline, memory, ready, row, context, leaf_problems.get(row['arena_policy']), profile)
            for term in row['cost']['terms']:
                key = term['key']
                query = term.pop('query')
                if cost_queries.setdefault(key, query) != query:
                    raise ValueError('one cost query key must have a single exact scope and context')
                if term['missing_terms']:
                    missing_costs.setdefault(key, dict(role=term['role'],
                        reasons=term['missing_terms'], applicable_case_ids=[]))['applicable_case_ids'].append(row['id'])
            cases.append(row)
        decision = decide(cases)
        original_unpriced = cases[0]['cost']['predicted_ns'] is None
        queue = []
        for key, query in missing_costs.items():
            role = query['role']
            missing_model = role in ('original_region', 'dependency_handoffs', 'workspace_arena_lifecycle',
                                     'worker_start_and_join')
            queue.append(dict(key=key, **query,
                acquisition=('NEEDS_SOURCE_QUALIFIED_COMPOSITION' if missing_model else
                    'DEFER_UNTIL_ORIGINAL_BOUNDARY_COST_IS_QUALIFIED' if original_unpriced else
                    'ELIGIBLE_EXACT_COMPONENT_QUERY'),
                whole_program_variant_probe_authorized=False))
        report = dict(schema=RESOURCE_EVENT_DOMAIN, schema_version=1, source_inputs=inputs_ref,
            cost_profile=profile, source_evidence=list(source_refs),
            original_task_graph_sha256=semantic_sha256(source_ready['task_graph']),
            declared_scope='ONE_GUARDED_CALLER_PIPELINE_INVOCATION_WARM_SESSION',
            placement_scope='ONE_DETERMINISTIC_ELIGIBLE_SAME_NUMA_PHYSICAL_CORE_PAIR_AND_ITS_FIRST_CORE',
            source_and_dependency_arenas=True, source_order_and_physical_resource_constraints_replayed=True,
            affinity_pinning_does_not_establish_page_residency=True,
            cases=cases, decision=decision, decision_cost_frontier=queue,
            all_cases_retained=True, minimum_gain_fraction=dict(numerator=1, denominator=20),
            probes_run_by_planner=False, declared_ticks_used_as_ns=False, physical_optimality_proven=False,
            target_program_executed=False, complete_variants_timed=False, deployment_authorized=False)
        return report, dict(schema=RESOURCE_EVENT_QUERIES, schema_version=1, queries=cost_queries)


def solve_pipeline_resource_events(inputs, out, *, cost_profile=None, search_budget=100000):
    """Replay source events and construct original/native resource alternatives."""
    from .project_pipeline_temporal import INPUTS
    from scar.optimization.implementation_costs import SCHEMA as COSTS

    output = Path(out).resolve()
    if output.exists():
        raise ValueError('fresh resource-event domain output required')
    declaration = dict(schema=RESOURCE_EVENT_INPUTS, schema_version=1,
        source_inputs=_file_reference(inputs, INPUTS, 'inputs.json'),
        cost_profile=None if cost_profile is None else _file_reference(cost_profile, COSTS, 'profile.json'),
        search_budget_per_arena=search_budget)
    output.mkdir(parents=True)
    declared = _write(output/'domain-inputs.json', declaration)
    report, queries = _resource_event_report(declaration, output=output)
    report['declared_domain'] = declared
    report['cost_queries'] = _write(output/'cost-queries.json', queries)
    _write(output/'result.json', report)
    return report


def verify_pipeline_resource_event_domain(report):
    """Cold-replay source, CPU topology, arena cases and all scoped prices."""
    ref = _file_reference(report, RESOURCE_EVENT_DOMAIN, 'result.json')
    value = read_artifact(ref)
    declaration = read_artifact(value['declared_domain'])
    if (value.get('schema_version') != 1
            or (declaration.get('schema'), declaration.get('schema_version')) != (RESOURCE_EVENT_INPUTS, 1)):
        raise ValueError('versioned original resource-event domain and declaration required')
    expected, queries = _resource_event_report(declaration, stored=value['cases'])
    expected['declared_domain'] = value['declared_domain']
    if semantic_sha256(queries) != semantic_sha256(read_artifact(value['cost_queries'])):
        raise ValueError('retained cost queries differ from original scope and context replay')
    expected['cost_queries'] = value['cost_queries']
    if semantic_sha256(expected) != semantic_sha256(value):
        raise ValueError('resource-event domain differs from source, resource, arena or cost replay')
    read_artifact(ref)
    return value


@contextmanager
def open_pipeline_resource_event_case(domain, case_id=None):
    """Open an explicitly requested trial or the replayed cost-model choice."""
    from .project_pipeline_temporal import open_temporal_inputs

    ref = _file_reference(domain, RESOURCE_EVENT_DOMAIN, 'result.json')
    report = verify_pipeline_resource_event_domain(ref)
    if case_id is None:
        decision = report['decision']
        if decision['selection_decision'] != 'COST_SELECTED_EXPLICIT_CASE_ONLY':
            raise ValueError('resource-event costs have not selected a native case; explicit trial id required')
        case_id = decision['selected_case_id']
    row = next((row for row in report['cases'] if row['id'] == case_id), None)
    if row is None or row['legal_status'] != 'ADMITTED_EXPLICIT_TRIAL':
        raise ValueError('one source/resource-admitted native trial case is required')
    with open_temporal_inputs(row['inputs']) as (pipeline, memory, ready, target):
        yield ref, case_id, row, pipeline, memory, ready, target
    read_artifact(ref)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--projection', action='append', metavar='L2_RESULT')
    mode.add_argument('--resource-events', metavar='TEMPORAL_INPUTS')
    mode.add_argument('--check-resource-events', metavar='DOMAIN_RESULT')
    parser.add_argument('--out')
    parser.add_argument('--cost-profile', metavar='EXACT_IMPLEMENTATION_COST_PROFILE')
    parser.add_argument('--search-budget', type=int, default=100000)
    args = parser.parse_args(argv)
    if args.check_resource_events:
        if args.out or args.cost_profile:
            parser.error('a saved domain already fixes its output and cost profile')
        result = verify_pipeline_resource_event_domain(args.check_resource_events)
        location = _file_reference(args.check_resource_events, RESOURCE_EVENT_DOMAIN, 'result.json')['uri']
    elif not args.out:
        parser.error('--out is required when publishing a domain')
    elif args.resource_events:
        result = solve_pipeline_resource_events(args.resource_events, args.out,
            cost_profile=args.cost_profile, search_budget=args.search_budget)
        location = str(Path(args.out)/'result.json')
    else:
        if args.cost_profile:
            parser.error('--cost-profile applies to the resource-event domain')
        result = solve_pipeline_temporal(args.projection, args.out, search_budget=args.search_budget)
        location = str(Path(args.out)/'result.json')
    decision = result['decision']
    print(json.dumps(dict(status=decision.get('status', decision['selection_decision']), case_count=len(result['cases']),
        selected_case_id=decision['selected_case_id'], result=location)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

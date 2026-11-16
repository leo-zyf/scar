"""L3 finite layout-domain makespan search over replayed L2 publications.

No source, mathematical backend, execution certificate or native host is
generated. Unknown cases stay in the domain and prevent a closed selection.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.pipeline-temporal-domain'
INPUTS_SCHEMA = 'scar.pipeline-temporal-domain-inputs'


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--projection', action='append', required=True, metavar='L2_RESULT')
    parser.add_argument('--out', required=True)
    parser.add_argument('--search-budget', type=int, default=100000)
    args = parser.parse_args(argv)
    result = solve_pipeline_temporal(args.projection, args.out, search_budget=args.search_budget)
    print(json.dumps(dict(status=result['decision']['status'], case_count=result['case_count'],
        selected_case_id=result['decision']['selected_case_id'], result=str(Path(args.out)/'result.json'))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

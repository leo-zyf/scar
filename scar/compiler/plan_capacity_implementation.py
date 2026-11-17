"""Source-replay capacity alternatives, solve their explicit model and export trials."""
import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import digest, problem_document
from scar.optimization.capacity_implementation import COST_SCHEMA, derive_capacity_problem, verify_capacity_problem
from scar.optimization.joint_planner import solve_implementations, verify_execution_plan

SCHEMA = 'scar.capacity-implementation-preparation'


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _reference(path, schema):
    path = Path(path).resolve()
    return artifact_reference(path/'result.json' if path.is_dir() else path, schema)


def _pins():
    root = Path(__file__).resolve().parents[1]
    return {name:artifact_reference(root/name, 'scar.python-source') for name in (
        'compiler/plan_capacity_implementation.py', 'optimization/capacity_implementation.py',
        'ir/implementation_plan.py', 'optimization/joint_planner.py',
        'compiler/session_capacity.py', 'compiler/session_outer.py')}


def _write(path, document, *, replace=False):
    with path.open('w' if replace else 'x') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, document['schema'])


def _inputs(preparation, steps, epochs, costs):
    from scar.compiler.session_capacity import replay as replay_capacity
    report, _, _, check = replay_capacity(preparation['uri'])
    _need(report == read_artifact(preparation), 'capacity preparation changed during full source replay')
    cost_input = None if costs is None else dict(reference=costs, document=read_artifact(costs))
    cases = []
    for ref in report['capacity_plans']:
        source = read_artifact(ref)
        kwargs = dict(capacity_plan_ref=ref, steps=steps, epochs=epochs, costs=cost_input)
        problem = derive_capacity_problem(source, **kwargs)
        verify_capacity_problem(source, problem, **kwargs)
        cases.append((ref, source, problem, kwargs))
    _need(bool(cases), 'at least one original source capacity region required')
    return report, cases, check


def _verify_retained(problem, plan):
    """Check feasibility or unresolved-domain preservation, without rerunning CP-SAT."""
    from scar.optimization.joint_planner import _frontiers
    _need(plan.get('schema') == 'scar.implementation-execution-plan'
          and plan.get('schema_version') == 1 and plan.get('problem_sha256') == digest(problem)
          and plan.get('baseline_only') is False,
          'retained model plan belongs to another original problem/domain')
    frontiers = _frontiers(problem, False)
    _need(plan.get('frontiers') == frontiers and plan.get('domain_complete') == (not bool(frontiers)),
          'unknown implementation alternatives were discarded')
    _need(plan.get('execution_authorized') is False and plan.get('deployment_authorized') is False
          and plan.get('physical_optimality_proven') is False and plan.get('target_program_executed') is False,
          'model plan cannot acquire physical/runtime authority')
    if plan.get('selected'):
        return verify_execution_plan(problem, plan)
    _need(plan.get('makespan_ns') is None and not plan.get('transfers') and not plan.get('memory_leases')
          and plan.get('model_optimality_proven') is False,
          'unresolved/infeasible model cannot acquire a free or selected implementation')
    if frontiers:
        _need(plan.get('status') == 'UNRESOLVED' and plan.get('model_lower_bound_ns') is None
              and plan.get('relative_gap') is None, 'unknown costs cannot become model bounds or optimality')
    else:
        _need(plan.get('status') in ('UNRESOLVED', 'INFEASIBLE_WITHIN_MODEL'),
              'empty complete-domain plan must remain unresolved or retained infeasible')
    return dict(feasible=False, unresolved_cost_domain=bool(frontiers),
        infeasibility_independently_proven=False, solver_optimality_independently_proven=False)


def _policy(problem, plan):
    if (not plan['domain_complete'] or not plan['selected']
            or plan['status'] not in ('OPTIMAL_WITHIN_MODEL', 'FEASIBLE_WITHIN_MODEL')):
        return None
    selected = plan['selected']
    _need(len(problem.operations) == len(selected) == 1, 'one original region per capacity model case required')
    row = selected[0]; op = problem.operations[0]
    option = next(option for option in op.options if option.id == row['option'])
    return dict(operation=op.id, option=option.id, lowering=option.lowering)


def _summary(cases):
    unresolved = any(row['solver_status'] == 'UNRESOLVED' or not row['domain_complete'] for row in cases)
    ready = all(row['selected_policy'] is not None for row in cases)
    return dict(status='UNRESOLVED' if unresolved else 'EXPLICIT_MODEL_TRIAL_PLAN_READY' if ready
        else 'NO_COMPLETE_FEASIBLE_MODEL_PLAN', regions=len(cases),
        independent_region_models=True, complete_program_makespan_proven=False,
        unknown_alternatives=sum(row['unknown_alternatives'] for row in cases),
        feasible_regions=sum(row['verification']['feasible'] for row in cases),
        trial_policy_ready=ready, physical_optimality_proven=False)


def _export(preparation, capacity_report, destination, cases, problem_ref, plan_ref, costs, steps, epochs):
    from scar.runtime.session_scope_bundle import inspect_scope_bundle, ADMISSION
    policies = [row['selected_policy'] for row in cases]
    _need(all(policies) and len({p['lowering'] for p in policies}) == 1,
          'one complete known uniform runtime policy is required for this bundle exporter')
    lowering = policies[0]['lowering']
    if lowering == 'RETAIN_PRIVATE_CAPACITY_FULL_SNAPSHOT':
        from scar.compiler.session_capacity import export
        exported = export(preparation['uri'], destination)
    elif lowering == 'FRESH_ISOLATED_NATIVE':
        from scar.compiler.session_outer import export
        exported = export(capacity_report['outer_preparation']['uri'], destination)
    else:
        raise ValueError('selected model has no existing capacity runtime exporter')
    directory = Path(exported['bundle'])
    scope_ref = artifact_reference(directory/'scope.json', 'scar.workspace-session-scope')
    _need(read_artifact(scope_ref)['unqualified_kernel_policy'] == lowering,
          'exported runtime policy differs from the selected implementation')
    binding = dict(problem=problem_ref, plan=plan_ref, preparation=preparation,
        costs=costs, selected_policies=policies, steps=steps, epochs=epochs,
        performance_parameters_are_source_loop_counts=False,
        performance_parameters_are_runtime_guards=False,
        explicit_trial_only=True, measured_profitability_proven=False,
        model_optimality_independently_reproved=False, deployment_authorized=False)
    admission_ref = artifact_reference(directory/'scope-admission.json', ADMISSION)
    admission = read_artifact(admission_ref)
    admission['implementation_planning'] = binding
    admission_ref = _write(directory/'scope-admission.json', admission, replace=True)
    inspect_scope_bundle(directory, expected_admission_sha256=admission_ref['sha256'])
    exported.update(expected_admission_sha256=admission_ref['sha256'],
        implementation_planning=binding, explicit_trial_only=True)
    exported_ref = _write(directory/'scope-export.json', exported, replace=True)
    return dict(bundle=str(directory), metadata=exported_ref, scope=scope_ref,
        admission=admission_ref, expected_admission_sha256=admission_ref['sha256'],
        selected_lowering=lowering, implementation_planning=binding,
        native_recompiled=False, native_math_changed=False, deployment_authorized=False)


def _check_export(exported, cases, preparation, capacity_report, source_check,
                  problem_ref, plan_ref, costs, steps, epochs):
    from scar.runtime.session_scope_bundle import inspect_scope_bundle
    admitted = inspect_scope_bundle(exported['bundle'],
        expected_admission_sha256=exported['expected_admission_sha256'])
    policies = [row['selected_policy'] for row in cases]
    _need(all(policies) and len({p['lowering'] for p in policies}) == 1,
          'exported runtime cannot use an incomplete or mixed model selection')
    binding = dict(problem=problem_ref, plan=plan_ref, preparation=preparation,
        costs=costs, selected_policies=policies, steps=steps, epochs=epochs,
        performance_parameters_are_source_loop_counts=False,
        performance_parameters_are_runtime_guards=False,
        explicit_trial_only=True, measured_profitability_proven=False,
        model_optimality_independently_reproved=False, deployment_authorized=False)
    expected_scope = (capacity_report['scope'] if policies[0]['lowering']
        == 'RETAIN_PRIVATE_CAPACITY_FULL_SNAPSHOT' else
        read_artifact(capacity_report['outer_preparation'])['scope'])
    _need(admitted['spec']['unqualified_kernel_policy'] == policies[0]['lowering']
          and admitted['spec'] == read_artifact(expected_scope)
          and admitted['admission']['session_admission_sha256'] == source_check['admission']['sha256']
          and admitted['parent']['admission'] == read_artifact(source_check['admission'])
          and admitted['admission'].get('implementation_planning') == binding
          and read_artifact(exported['metadata'])['implementation_planning'] == binding
          and read_artifact(exported['metadata'])['expected_admission_sha256'] == exported['expected_admission_sha256']
          and read_artifact(exported['admission']) == admitted['admission']
          and read_artifact(exported['scope']) == admitted['spec']
          and exported['implementation_planning'] == binding
          and exported['selected_lowering'] == policies[0]['lowering'],
          'export metadata/source admission differs from exact model selection')


def _report(preparation, costs, steps, epochs, problem_ref, plan_ref, cases, export_requested, exported):
    return dict(schema=SCHEMA, schema_version=1, preparation=preparation, costs=costs,
        steps=steps, epochs=epochs, problem=problem_ref, plan=plan_ref, cases=cases,
        implementations=_pins(), summary=_summary(cases), export_requested=export_requested,
        exported=exported, costs_are_measurements=False,
        performance_parameters_not_source_loop_authority=True,
        check_scope='SOURCE_PROBLEM_AND_RETAINED_PLAN_FEASIBILITY_NOT_OPTIMALITY_REPROOF',
        original_source_control_retained=True, source_replacement_authorized=False,
        native_compiled=False, target_program_executed=False,
        automatic_deployment_authorized=False, default_backend='NO_OP', deployment_authorized=False)


def prepare(preparation, out, *, steps, epochs, costs=None, export_directory=None):
    output = Path(out).resolve()
    _need(not output.exists(), 'fresh capacity implementation output required')
    export_path = None if export_directory is None else str(Path(export_directory).resolve())
    _need(export_path is None or not Path(export_path).exists(), 'fresh trial runtime output required')
    source = _reference(preparation, 'scar.session-capacity-preparation')
    cost_ref = None if costs is None else _reference(costs, COST_SCHEMA)
    capacity_report, original, source_check = _inputs(source, steps, epochs, cost_ref)
    output.mkdir(parents=True, exist_ok=False)
    cases = []; problem_refs = []; plan_refs = []
    multiple = len(original) > 1
    for index, (ref, _, problem, _) in enumerate(original):
        prefix = str(index)+'-' if multiple else ''
        plan = solve_implementations(problem)
        verification = _verify_retained(problem, plan)
        problem_ref = _write(output/(prefix+'problem.json'), problem_document(problem))
        plan_ref = _write(output/(prefix+'plan.json'), plan)
        problem_refs.append(problem_ref); plan_refs.append(plan_ref)
        cases.append(dict(capacity_plan=ref, problem=problem_ref, plan=plan_ref,
            solver_status=plan['status'], domain_complete=plan['domain_complete'],
            unknown_alternatives=len(plan['frontiers']), verification=verification,
            selected_policy=_policy(problem, plan)))
    problem_ref = _write(output/'problem.json', dict(schema='scar.capacity-independent-problems',
        schema_version=1, problems=problem_refs)) if multiple else problem_refs[0]
    plan_ref = _write(output/'plan.json', dict(schema='scar.capacity-independent-plans',
        schema_version=1, plans=plan_refs, summary=_summary(cases))) if multiple else plan_refs[0]
    exported = None
    if (export_path and all(row['selected_policy'] for row in cases)
            and len({row['selected_policy']['lowering'] for row in cases}) == 1):
        exported = _export(source, capacity_report, export_path, cases, problem_ref, plan_ref,
            cost_ref, steps, epochs)
        _check_export(exported, cases, source, capacity_report, source_check,
            problem_ref, plan_ref, cost_ref, steps, epochs)
    report = _report(source, cost_ref, steps, epochs, problem_ref, plan_ref, cases, export_path, exported)
    _write(output/'result.json', report)
    return report


def replay(receipt):
    reference = _reference(receipt, SCHEMA); report = read_artifact(reference)
    capacity_report, original, source_check = _inputs(report['preparation'], report['steps'], report['epochs'], report['costs'])
    _need(len(original) == len(report['cases']), 'original capacity region domain changed')
    cases = []
    for (ref, source, problem, kwargs), retained in zip(original, report['cases']):
        _need(retained['capacity_plan'] == ref, 'retained case changed original source region')
        document = read_artifact(retained['problem'])
        verify_capacity_problem(source, document, **kwargs)
        plan = read_artifact(retained['plan']); verification = _verify_retained(problem, plan)
        cases.append(dict(capacity_plan=ref, problem=retained['problem'], plan=retained['plan'],
            solver_status=plan['status'], domain_complete=plan['domain_complete'],
            unknown_alternatives=len(plan['frontiers']), verification=verification,
            selected_policy=_policy(problem, plan)))
    if len(cases) == 1:
        _need(report['problem'] == cases[0]['problem'] and report['plan'] == cases[0]['plan'],
              'single source region must retain its direct problem/plan references')
    else:
        _need(read_artifact(report['problem']) == dict(schema='scar.capacity-independent-problems',
            schema_version=1, problems=[r['problem'] for r in cases])
            and read_artifact(report['plan']) == dict(schema='scar.capacity-independent-plans',
            schema_version=1, plans=[r['plan'] for r in cases], summary=_summary(cases)),
            'independent source cases cannot be discarded or merged into a fictitious program schedule')
    if report['exported'] is not None:
        _check_export(report['exported'], cases, report['preparation'], capacity_report, source_check,
            report['problem'], report['plan'], report['costs'], report['steps'], report['epochs'])
    expected = _report(report['preparation'], report['costs'], report['steps'], report['epochs'],
        report['problem'], report['plan'], cases, report['export_requested'], report['exported'])
    _need(digest(report) == digest(expected), 'capacity model publication differs from full source/problem/plan replay')
    read_artifact(reference)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--preparation'); source.add_argument('--check')
    parser.add_argument('--steps', type=int); parser.add_argument('--epochs', type=int)
    parser.add_argument('--costs'); parser.add_argument('--out'); parser.add_argument('--export', dest='export_directory')
    args = parser.parse_args(argv)
    if args.check:
        if any(value is not None for value in (args.steps, args.epochs, args.costs, args.out, args.export_directory)):
            parser.error('check consumes only the retained capacity implementation receipt')
        report = replay(args.check)
    else:
        if args.steps is None or args.epochs is None or not args.out:
            parser.error('preparation requires --steps, --epochs and --out')
        report = prepare(args.preparation, args.out, steps=args.steps, epochs=args.epochs,
            costs=args.costs, export_directory=args.export_directory)
    print(json.dumps(dict(summary=report['summary'], problem=report['problem'], plan=report['plan'],
        exported=report['exported'], deployment_authorized=False), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

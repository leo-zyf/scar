"""Replay semantic time, implementation refinements and original resource bindings."""
import argparse
import json
from pathlib import Path
import sys

from scar.analysis.schedule_time import attach_task_time
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.temporal_resources import projection_document, projection_from_dict
from scar.ir.versioned_schedule import TemporalOccupancyKind, TemporalScopeKind, validate_versioned_schedule
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.schedule import schedule_document, schedule_from_dict

SCHEMA = 'scar.temporal-resource-preparation'
TIME_SCHEMA = 'scar.semantic-time-preparation'
IMPLEMENTATION_SCHEMA = 'scar.implementation-time-preparation'
WORKSPACE_TIME = 'scar.workspace-time-preparation'
WORKSPACE_DEMANDS = 'scar.workspace-demand-time-preparation'
SERIAL_AFFINITY = 'SOURCE_PROVEN_SERIAL_AFFINITY'
SERIAL_SCOPE = 'SOURCE_PROVEN_SERIAL_SCOPE_DEMANDS'
SCOPE_CLEANUP = 'SOURCE_PROVEN_SCOPE_CLEANUP'


def time_schedule_reference(path):
    """Route admitted time formats before their independent source replay."""
    from scar.ir.execution_recipe import _bytes, _json
    path = Path(path).resolve()
    schema = _json(_bytes(path)).get('schema')
    if schema not in (TIME_SCHEMA, IMPLEMENTATION_SCHEMA, WORKSPACE_TIME, WORKSPACE_DEMANDS):
        raise ValueError('a supported source-replayed time preparation required')
    return artifact_reference(path, schema)


def _loop_context(reference, schedule, region_reference):
    from scar.analysis.build import BuildSession
    from scar.compiler.project_pipeline_temporal import _provider
    from scar.compiler.frontend.stateful import read_stateful_loop_proof
    from scar.compiler.stateful_model import document
    raw = read_artifact(reference)
    if (raw.get('schema'), raw.get('schema_version')) != ('scar.g1-current-source-loop-reference', 1):
        raise ValueError('a current source loop reference is required')
    for name in ('program', 'current_proof', 'step'):
        read_artifact(raw[name])
    manifest = Path(raw['program']['uri'])
    if raw['program'] != artifact_reference(manifest.parent/'manifest.json', 'scar.program-build'):
        raise ValueError('original program manifest reference required')
    with BuildSession(manifest.parent) as session:
        _, provider = _provider(session, raw['provider'])
        proof = read_stateful_loop_proof(raw['current_proof']['uri'], view=session.view, provider=provider)
        if (schedule.region_id != proof.step.id or proof.step_reference != raw['step']
                or semantic_sha256(read_artifact(region_reference)) != semantic_sha256(document(proof.step))):
            raise ValueError('loop body must be this exact source-proven stateful step')
        refs = ({'artifact': reference, 'locator': '#/current_proof'},
                {'artifact': raw['current_proof'], 'locator': '#/origins'},
                {'artifact': raw['current_proof'], 'locator': '#/parameter_map'})
        return refs, proof.trip_count_parameter


def _adapt(kind, reference, loop_reference=None):
    source_refs = ({'artifact': reference, 'locator': '#'},)
    if kind == 'pipeline':
        if loop_reference is not None:
            raise ValueError('source-proven loop adapter currently requires a stateful recipe')
        from scar.compiler.project_pipeline_temporal import open_temporal_inputs
        with open_temporal_inputs(reference) as (_, _, ready, _):
            original = schedule_from_dict(ready['task_graph'])
            demands = {task.id: ready['call_resource_demands'][task.attrs['call_id']]
                       for task in original.tasks if 'call_id' in task.attrs}
            result = attach_task_time(original, execution_demands=demands,
                resource_capacities=ready['resource_capacities'], source_refs=source_refs)
    elif kind == 'stateful_recipe':
        from scar.optimization.stateful_recipe import verify_stateful_recipe
        recipe = verify_stateful_recipe(reference['uri'])
        # Verify the retained bytes, not just a file with the same path.
        if recipe != read_artifact(reference):
            raise ValueError('stateful recipe reference changed during source replay')
        original = schedule_from_dict(recipe['task_graph'])
        loop_refs, trip_symbol = (), None
        if loop_reference is not None:
            loop_refs, trip_symbol = _loop_context(loop_reference, original, recipe['artifacts']['region'])
        result = attach_task_time(original, source_refs=source_refs,
            loop_refs=loop_refs, trip_count_symbol=trip_symbol)
    else:
        raise ValueError('unsupported original time adapter')
    return result


def _summary(schedule):
    result = validate_versioned_schedule(schedule)
    result['source_task_graph_executable'] = result.pop('executable')
    result['source_budget_scope'] = schedule.budget_scope
    result['source_budget_bytes'] = schedule.budget_bytes
    result['physical_memory_domain_aggregation'] = 'NOT_YET_PROJECTED'
    result['external_slots_have_no_invented_allocation_or_free'] = True
    result['storage_ids_in_loop_body'] = ('PER_ITERATION_TEMPLATE_NOT_CROSS_ITERATION_REUSE'
        if any(scope.iteration_symbol for scope in schedule.temporal.scopes) else 'NOT_A_LOOP')
    return result


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, value['schema'])


def _observed_effects(kind, reference, resource_scopes):
    if kind != 'stateful_recipe':
        raise ValueError('observed effect order needs an original stateful recipe source')
    from scar.analysis.observed.boundaries import project_observed_effect_order
    from scar.analysis.source_resource_scopes import SCHEMA as SCOPES
    scopes = read_artifact(resource_scopes)
    path = Path(resource_scopes['uri'])
    if path.name != 'result.json' or resource_scopes['schema'] != SCOPES:
        raise ValueError('original resource scope publication required')
    region = read_artifact(read_artifact(reference)['artifacts']['region'])
    order = project_observed_effect_order(path.parent, region['source'])
    if read_artifact(resource_scopes) != scopes:
        raise ValueError('resource scope publication changed during time projection')
    return dict(resource_scopes=resource_scopes, order=order)


def _time_report(kind, reference, loop_ref, schedule_ref, summary, effects):
    receipt = dict(schema=TIME_SCHEMA, schema_version=1, source_kind=kind, source=reference,
        loop_source=loop_ref, schedule=schedule_ref, summary=summary, target_program_executed=False,
        solver_invoked=False, native_compiled=False, performance_measured=False,
        deployment_authorized=False)
    if effects is not None:
        receipt['observed_effects'] = effects
    return receipt


def prepare_time_schedule(kind, source, out, *, loop_reference=None, resource_scopes=None):
    schema = 'scar.pipeline-temporal-inputs' if kind == 'pipeline' else 'scar.execution-recipe'
    reference = artifact_reference(Path(source).resolve(), schema)
    loop_ref = (artifact_reference(Path(loop_reference).resolve(), 'scar.g1-current-source-loop-reference')
                if loop_reference is not None else None)
    schedule = _adapt(kind, reference, loop_ref)
    effects = None
    if resource_scopes is not None:
        from scar.analysis.source_resource_scopes import SCHEMA as SCOPES
        effects = _observed_effects(kind, reference, artifact_reference(
            Path(resource_scopes).resolve()/'result.json', SCOPES))
    directory = Path(out).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    schedule_ref = _write(directory/'schedule.json', schedule_document(schedule))
    receipt = _time_report(kind, reference, loop_ref, schedule_ref, _summary(schedule), effects)
    _write(directory/'result.json', receipt)
    return receipt


def _replay_semantic_time(reference):
    receipt = read_artifact(reference)
    if (receipt.get('schema'), receipt.get('schema_version')) != (TIME_SCHEMA, 1):
        raise ValueError('semantic time receipt required')
    saved = schedule_from_dict(read_artifact(receipt['schedule']))
    kind, source, loop_ref = receipt['source_kind'], receipt['source'], receipt['loop_source']
    replay = _adapt(kind, source, loop_ref)
    effects = None
    if 'observed_effects' in receipt:
        effects = _observed_effects(kind, source, receipt['observed_effects']['resource_scopes'])
    expected = _time_report(kind, source, loop_ref, receipt['schedule'], _summary(replay), effects)
    if saved != replay or receipt != expected:
        raise ValueError('semantic time graph differs from complete original source replay')
    read_artifact(reference, json_document=False)
    return receipt, saved


def _derive_implementation(reference, *, serial_demands=False, scope_cleanup=False, source_rule=None):
    from scar.analysis.implementation_envelope import derive_stateful_envelope
    if reference.get('schema') != TIME_SCHEMA:
        raise ValueError('implementation refinement requires the original unrefined time receipt')
    base, original = _replay_semantic_time(reference)
    if base['source_kind'] != 'stateful_recipe':
        raise ValueError('current implementation envelope contract is the existing stateful CPU backend')
    recipe = read_artifact(base['source'])
    if any(scope.kind is TemporalScopeKind.LOOP for scope in original.temporal.scopes):
        raise ValueError('standalone method envelope cannot model a repeat adapter; preserve the original shared loop')
    if scope_cleanup:
        from scar.analysis.implementation_scope import derive_stateful_scope_envelope
        refined = derive_stateful_scope_envelope(original, recipe)
    else:
        refined = derive_stateful_envelope(original, recipe)
    activities = {row.id: row for row in original.temporal.activities}
    demands = {}
    for row in original.temporal.occupancies:
        if row.kind is TemporalOccupancyKind.EXECUTION_DEMAND:
            task = activities[row.activity].task
            demands.setdefault(task, {})[row.resource] = row.units
    schedule = attach_task_time(refined, execution_demands=demands,
        resource_capacities=original.temporal.resource_capacities,
        source_refs=original.temporal.source_refs)
    if scope_cleanup:
        from scar.analysis.control_milestone import derive_scope_control_milestone
        schedule = derive_scope_control_milestone(schedule, recipe)
    if serial_demands:
        target = recipe['artifacts']['target_snapshot']
        read_artifact(target)
        if scope_cleanup:
            from scar.analysis.scope_demands import derive_serial_scope_demands
            schedule = derive_serial_scope_demands(schedule, recipe, Path(target['uri']).parent, source_rule=source_rule)
        else:
            from scar.analysis.scope_demands import derive_serial_affinity_demands
            schedule = derive_serial_affinity_demands(schedule, recipe, Path(target['uri']).parent)
    read_artifact(reference, json_document=False)
    return base, schedule, _summary(schedule)


def _implementation_report(base_reference, base, schedule_ref, summary, *, serial_demands=False, scope_cleanup=False):
    result = dict(schema=IMPLEMENTATION_SCHEMA, schema_version=1, base_time=base_reference,
        source_kind=base['source_kind'], source=base['source'], loop_source=base['loop_source'],
        schedule=schedule_ref, summary=summary,
        layer='MODEL_EXISTING_IMPLEMENTATION_ENVELOPE',
        target_program_executed=False, solver_invoked=False, native_compiled=False,
        performance_measured=False, deployment_authorized=False)
    if serial_demands:
        result['resource_refinement'] = SERIAL_SCOPE if scope_cleanup else SERIAL_AFFINITY
    if scope_cleanup:
        result['implementation_refinement'] = SCOPE_CLEANUP
    return result


def prepare_implementation_time(time_schedule, out, *, serial_demands=False, scope_cleanup=False):
    reference = time_schedule_reference(time_schedule)
    base, schedule, summary = _derive_implementation(reference, serial_demands=serial_demands, scope_cleanup=scope_cleanup)
    directory = Path(out).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    result = _implementation_report(reference, base,
        _write(directory/'schedule.json', schedule_document(schedule)), summary,
        serial_demands=serial_demands, scope_cleanup=scope_cleanup)
    _write(directory/'result.json', result)
    return result


def _replay_implementation_time(reference):
    from scar.analysis.scope_demands import SCHEMA as SCOPE_PROOF, demand_proof

    result = read_artifact(reference)
    if (result.get('schema'), result.get('schema_version')) != (IMPLEMENTATION_SCHEMA, 1):
        raise ValueError('implementation time preparation receipt required')
    if result.get('resource_refinement') not in (None, SERIAL_AFFINITY, SERIAL_SCOPE):
        raise ValueError('unknown implementation resource refinement')
    if result.get('implementation_refinement') not in (None, SCOPE_CLEANUP):
        raise ValueError('unknown source implementation scope refinement')
    scope_cleanup = result.get('implementation_refinement') == SCOPE_CLEANUP
    serial_demands = result.get('resource_refinement') in (SERIAL_AFFINITY, SERIAL_SCOPE)
    if serial_demands and scope_cleanup != (result['resource_refinement'] == SERIAL_SCOPE):
        raise ValueError('scope and legacy resource refinements cannot be mixed')
    saved = schedule_from_dict(read_artifact(result['schedule']))
    proof = demand_proof(saved, SCOPE_PROOF) if serial_demands and scope_cleanup else None
    base, replay, summary = _derive_implementation(result['base_time'], serial_demands=serial_demands,
        scope_cleanup=scope_cleanup, source_rule=None if proof is None else proof[1]['source_rule'])
    if saved != replay or result != _implementation_report(result['base_time'], base, result['schedule'], summary,
            serial_demands=serial_demands, scope_cleanup=scope_cleanup):
        raise ValueError('implementation envelope differs from original source/backend/phase replay')
    read_artifact(reference, json_document=False)
    return result, saved


def _read_workspace_receipt(reference):
    if reference['schema'] == WORKSPACE_TIME:
        from scar.compiler.project_workspace_time import read_workspace_time
        receipt = read_workspace_time(reference['uri'])
    else:
        from scar.compiler.project_workspace_demands import read_workspace_demands
        receipt = read_workspace_demands(reference['uri'])
    if receipt != read_artifact(reference):
        raise ValueError('original workspace time receipt changed during replay')
    return receipt


def replay_time_schedule(source):
    """Verify the retained time receipt and return its typed schedule."""
    reference = source if isinstance(source, dict) else time_schedule_reference(source)
    if reference['schema'] == IMPLEMENTATION_SCHEMA:
        return _replay_implementation_time(reference)
    if reference['schema'] in (WORKSPACE_TIME, WORKSPACE_DEMANDS):
        receipt = _read_workspace_receipt(reference)
        return receipt, schedule_from_dict(read_artifact(receipt['schedule']))
    return _replay_semantic_time(reference)


def read_time_schedule(path):
    reference = time_schedule_reference(path)
    if reference['schema'] in (WORKSPACE_TIME, WORKSPACE_DEMANDS):
        return _read_workspace_receipt(reference)
    return replay_time_schedule(reference)[0]


def read_implementation_time(path):
    return _replay_implementation_time(artifact_reference(Path(path).resolve(), IMPLEMENTATION_SCHEMA))[0]


def _derive_resources(time_reference):
    from scar.analysis.software_resources import derive_temporal_resources
    receipt, schedule = replay_time_schedule(time_reference)
    context = dict(caller_affinity_cpu_ids=None, recipe_binding_cpu_ids=None,
        source_refs=({'artifact': time_reference, 'locator': '#/source'},))
    if receipt['source_kind'] == 'pipeline':
        inputs = read_artifact(receipt['source'])
        ready = read_artifact(inputs['ready'])
        target = ready['target']
        context['caller_affinity_cpu_ids'] = ready['caller_affinity_cpu_ids']
    elif receipt['source_kind'] == 'stateful_workspace':
        preparation = read_artifact(receipt['source'])
        if ((preparation.get('schema'), preparation.get('schema_version'))
                != ('scar.workspace-preparation', 1)
                or preparation.get('workspace_plan') != receipt['workspace_plan']):
            raise ValueError('workspace time must retain its exact source preparation and shared workspace plan')
        target = receipt['target']
        plan = read_artifact(preparation['step_plan'])
        # The shared capacity lifetime is already in the v4 source graph.
        # This selected compute domain proves nothing about the caller's
        # outer allocator/releaser or an unknown per-iteration host service.
        context['recipe_binding_cpu_ids'] = plan['resource_binding']['cpu_ids']
        context['source_refs'] = (*context['source_refs'],
            {'artifact': receipt['source'], 'locator': '#/step_plan'},
            {'artifact': receipt['workspace_plan'], 'locator': '#/slots'})
    elif receipt['source_kind'] == 'stateful_recipe':
        recipe = read_artifact(receipt['source'])
        target = recipe['artifacts']['target_snapshot']
        plan = read_artifact(recipe['artifacts']['stage_plan'])
        # This is a chosen compute domain. Host allocations/guards may execute
        # before pinning and must not be assigned to it without source evidence.
        context['recipe_binding_cpu_ids'] = plan['resource_binding']['cpu_ids']
    else:
        raise ValueError('unsupported original temporal resource source')
    context['target_reference'] = target
    read_artifact(target)
    projection = derive_temporal_resources(schedule, Path(target['uri']).parent,
        source_context=context)
    read_artifact(time_reference, json_document=False)
    return projection


def _resource_summary(projection):
    return dict(execution_bindings=len(projection.executions),
        exact_thread_bindings=sum(row.exact_thread is not None for row in projection.executions),
        unknown_execution_units=sum(row.units is None for row in projection.executions),
        activities_without_a_known_cpu_domain=sum(not row.candidate_threads for row in projection.executions),
        owned_slot_bindings=len(projection.storages),
        exact_numa_placements=sum(row.exact_memory is not None for row in projection.storages),
        original_budget_bytes=projection.budget.bytes, original_budget_scope=projection.budget.scope,
        unresolved=len(projection.unresolved), physical_resource_feasibility_proven=False,
        inventory_totals_used_as_available_budgets=False)


def _resource_report(time_reference, projection_ref, projection):
    return dict(schema=SCHEMA, schema_version=1, time_schedule=time_reference,
        projection=projection_ref, summary=_resource_summary(projection),
        target_program_executed=False, solver_invoked=False, native_compiled=False,
        performance_measured=False, deployment_authorized=False)


def prepare_resource_bindings(time_schedule, out):
    reference = time_schedule_reference(time_schedule)
    projection = _derive_resources(reference)
    directory = Path(out).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    projection_ref = _write(directory/'resources.json', projection_document(projection))
    report = _resource_report(reference, projection_ref, projection)
    _write(directory/'result.json', report)
    return report


def read_resource_bindings(path):
    reference = artifact_reference(Path(path).resolve(), SCHEMA)
    report = read_artifact(reference)
    if (report.get('schema'), report.get('schema_version')) != (SCHEMA, 1):
        raise ValueError('temporal resource preparation receipt required')
    saved = projection_from_dict(read_artifact(report['projection']))
    replay = _derive_resources(report['time_schedule'])
    expected = _resource_report(report['time_schedule'], report['projection'], replay)
    if projection_document(saved) != projection_document(replay) or report != expected:
        raise ValueError('resource binding differs from original time/source/hardware replay')
    read_artifact(reference, json_document=False)
    return report


def _time_cli(argv):
    parser = argparse.ArgumentParser(description='Prepare and replay semantic time from original sources.')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--pipeline-inputs')
    inputs.add_argument('--recipe')
    inputs.add_argument('--check')
    parser.add_argument('--loop-reference')
    parser.add_argument('--resource-scopes', help='retain same-capture order of source-bound write scopes')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check:
        if args.out or args.loop_reference or args.resource_scopes:
            parser.error('check consumes only the retained receipt')
        result = read_time_schedule(args.check)
    else:
        if not args.out:
            parser.error('time graph preparation requires --out')
        result = prepare_time_schedule('pipeline' if args.pipeline_inputs else 'stateful_recipe',
            args.pipeline_inputs or args.recipe, args.out, loop_reference=args.loop_reference,
            resource_scopes=args.resource_scopes)
    print(json.dumps(dict(schema=result['schema'], summary=result['summary'],
        deployment_authorized=False), sort_keys=True))
    return 0


def _implementation_cli(argv):
    parser = argparse.ArgumentParser(description='Prepare and replay a source-backed CPU implementation envelope.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--time-schedule')
    source.add_argument('--check')
    parser.add_argument('--out')
    parser.add_argument('--bind-serial-demands', action='store_true')
    parser.add_argument('--scope-cleanup', action='store_true')
    args = parser.parse_args(argv)
    if args.check:
        if args.out or args.bind_serial_demands or args.scope_cleanup:
            parser.error('check consumes only the saved implementation-time receipt')
        result = read_implementation_time(args.check)
    else:
        if not args.out:
            parser.error('implementation-time preparation requires --out')
        result = prepare_implementation_time(args.time_schedule, args.out,
            serial_demands=args.bind_serial_demands, scope_cleanup=args.scope_cleanup)
    print(json.dumps(dict(summary=result['summary'], schedule=result['schedule'],
        deployment_authorized=False), sort_keys=True))
    return 0


def _resource_cli(argv):
    parser = argparse.ArgumentParser(description='Bind or replay hardware resources. The time and implementation commands prepare earlier receipts.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--time-schedule')
    source.add_argument('--check')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check:
        if args.out:
            parser.error('check consumes only the saved resource receipt')
        result = read_resource_bindings(args.check)
    else:
        if not args.out:
            parser.error('resource binding preparation requires --out')
        result = prepare_resource_bindings(args.time_schedule, args.out)
    print(json.dumps(result, sort_keys=True))
    return 0


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    commands = {'time': _time_cli, 'implementation': _implementation_cli, 'resources': _resource_cli}
    if args and args[0] in commands:
        return commands[args[0]](args[1:])
    return _resource_cli(args)


if __name__ == '__main__':
    raise SystemExit(main())

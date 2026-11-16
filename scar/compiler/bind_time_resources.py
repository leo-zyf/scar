"""Bind semantic-time occupancies to the original hardware inventory and budgets."""
import argparse
import json
from pathlib import Path

from scar.compiler.time_schedule import read_time_schedule, time_schedule_reference
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.schedule import schedule_from_dict

SCHEMA = 'scar.temporal-resource-preparation'


def _derive(time_reference):
    from scar.analysis.temporal_resources import derive_temporal_resources
    receipt = read_time_schedule(time_reference['uri'])
    if receipt != read_artifact(time_reference):
        raise ValueError('original semantic-time receipt changed during replay')
    schedule = schedule_from_dict(read_artifact(receipt['schedule']))
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
    read_artifact(time_reference)
    return projection


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, value['schema'])


def _summary(projection):
    return dict(execution_bindings=len(projection.executions),
        exact_thread_bindings=sum(row.exact_thread is not None for row in projection.executions),
        unknown_execution_units=sum(row.units is None for row in projection.executions),
        activities_without_a_known_cpu_domain=sum(not row.candidate_threads for row in projection.executions),
        owned_slot_bindings=len(projection.storages),
        exact_numa_placements=sum(row.exact_memory is not None for row in projection.storages),
        original_budget_bytes=projection.budget.bytes, original_budget_scope=projection.budget.scope,
        unresolved=len(projection.unresolved), physical_resource_feasibility_proven=False,
        inventory_totals_used_as_available_budgets=False)


def prepare_resource_bindings(time_schedule, out):
    from scar.ir.temporal_resources import projection_document
    reference = time_schedule_reference(time_schedule)
    projection = _derive(reference)
    directory = Path(out).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(schema=SCHEMA, schema_version=1, time_schedule=reference,
        projection=_write(directory/'resources.json', projection_document(projection)),
        summary=_summary(projection),
        target_program_executed=False, solver_invoked=False, native_compiled=False,
        performance_measured=False, deployment_authorized=False)
    _write(directory/'result.json', report)
    return report


def read_resource_bindings(path):
    from scar.ir.temporal_resources import projection_document, projection_from_dict
    reference = artifact_reference(Path(path).resolve(), SCHEMA)
    report = read_artifact(reference)
    if (report.get('schema'), report.get('schema_version')) != (SCHEMA, 1):
        raise ValueError('temporal resource preparation receipt required')
    saved = projection_from_dict(read_artifact(report['projection']))
    replay = _derive(report['time_schedule'])
    expected = dict(schema=SCHEMA, schema_version=1, time_schedule=report['time_schedule'],
        projection=report['projection'], summary=_summary(replay), target_program_executed=False,
        solver_invoked=False, native_compiled=False, performance_measured=False,
        deployment_authorized=False)
    if projection_document(saved) != projection_document(replay) or report != expected:
        raise ValueError('resource binding differs from original time/source/hardware replay')
    read_artifact(reference)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
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


if __name__ == '__main__':
    raise SystemExit(main())

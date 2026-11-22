"""Attach semantic time to source-replayed CPU schedules; no target execution."""
import argparse
import json
from pathlib import Path

from scar.analysis.schedule_time import attach_task_time
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.versioned_schedule import validate_versioned_schedule
from scar.optimization.schedule import schedule_document, schedule_from_dict
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.semantic-time-preparation'


def time_schedule_reference(path):
    """Route admitted time formats before their independent source replay."""
    from scar.ir.execution_recipe import _bytes, _json
    path = Path(path).resolve()
    schema = _json(_bytes(path)).get('schema')
    if schema not in (SCHEMA, 'scar.implementation-time-preparation', 'scar.workspace-time-preparation',
                      'scar.workspace-demand-time-preparation'):
        raise ValueError('a supported source-replayed time preparation required')
    return artifact_reference(path, schema)


def _loop_context(reference, schedule, region_reference):
    from scar.analysis.build import BuildSession
    from scar.compiler.project_pipeline_temporal import _provider
    from scar.compiler.stateful_loop_frontend import read_stateful_loop_proof
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
    from scar.analysis.observed_boundaries import project_observed_effect_order
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


def prepare_time_schedule(kind, source, out, *, loop_reference=None, resource_scopes=None):
    schema = 'scar.pipeline-temporal-inputs' if kind == 'pipeline' else 'scar.execution-recipe'
    reference = artifact_reference(Path(source).resolve(), schema)
    loop_ref = (artifact_reference(Path(loop_reference).resolve(), 'scar.g1-current-source-loop-reference')
                if loop_reference is not None else None)
    schedule = _adapt(kind, reference, loop_ref)
    if resource_scopes is not None:
        from scar.analysis.source_resource_scopes import SCHEMA as SCOPES
        effects = _observed_effects(kind, reference, artifact_reference(
            Path(resource_scopes).resolve()/'result.json', SCOPES))
    directory = Path(out).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema=SCHEMA, schema_version=1, source_kind=kind, source=reference,
        loop_source=loop_ref, schedule=_write(directory/'schedule.json', schedule_document(schedule)),
        summary=_summary(schedule), target_program_executed=False, solver_invoked=False,
        native_compiled=False, performance_measured=False, deployment_authorized=False)
    if resource_scopes is not None:
        receipt['observed_effects'] = effects
    _write(directory/'result.json', receipt)
    return receipt


def read_time_schedule(path):
    reference = time_schedule_reference(path)
    if reference['schema'] == 'scar.workspace-demand-time-preparation':
        from scar.compiler.project_workspace_demands import read_workspace_demands
        return read_workspace_demands(path)
    if reference['schema'] == 'scar.workspace-time-preparation':
        from scar.compiler.project_workspace_time import read_workspace_time
        return read_workspace_time(path)
    if reference['schema'] == 'scar.implementation-time-preparation':
        from scar.compiler.refine_time_envelope import read_implementation_time
        return read_implementation_time(path)
    receipt = read_artifact(reference)
    if (receipt.get('schema'), receipt.get('schema_version')) != (SCHEMA, 1):
        raise ValueError('semantic time receipt required')
    saved = schedule_from_dict(read_artifact(receipt['schedule']))
    replay = _adapt(receipt['source_kind'], receipt['source'], receipt['loop_source'])
    expected = dict(schema=SCHEMA, schema_version=1, source_kind=receipt['source_kind'],
        source=receipt['source'], loop_source=receipt['loop_source'], schedule=receipt['schedule'],
        summary=_summary(replay), target_program_executed=False, solver_invoked=False,
        native_compiled=False, performance_measured=False, deployment_authorized=False)
    if 'observed_effects' in receipt:
        expected['observed_effects'] = _observed_effects(receipt['source_kind'], receipt['source'],
            receipt['observed_effects']['resource_scopes'])
    if saved != replay or receipt != expected:
        raise ValueError('semantic time graph differs from complete original source replay')
    read_artifact(reference)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
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


if __name__ == '__main__':
    raise SystemExit(main())

"""Cold-replay parent graphs, then publish bounded closed-window models."""
import argparse
import json
from pathlib import Path

from scar.analysis.closed_windows import derive_closed_windows
from scar.compiler.project_activity_time import replay_activity_time, SCHEMA as TIME_SCHEMA
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.closed_window_planner import plan_closed_window_models
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.closed-window-preparation'


def _derive(reference, search_budget):
    time_report, projection, costs, table, derivation, schedule, resources = replay_activity_time(reference['uri'])
    if time_report != read_artifact(reference):
        raise ValueError('qualified activity time changed during window projection')
    recipe = read_artifact(time_report['inputs']['source'])
    refs = (dict(role='activity_time', artifact=reference, locator='#'),
        *(dict(role=role, artifact=value, locator='#') for role, value in time_report['inputs'].items()))
    windows = derive_closed_windows(schedule, resources, recipe, source_refs=refs)
    plans = plan_closed_window_models(schedule, resources, table, projection, windows,
        search_budget=search_budget)
    read_artifact(reference)
    context = dict(schedule=schedule, resources=resources, recipe=recipe, table=table,
        activity_time_projection=projection, activity_time_report=time_report,
        costs=costs, derivation=derivation, source_refs=refs)
    return windows, plans, context


def _write(path, document):
    with path.open('x') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, document['schema'])


def _report(reference, windows_ref, plans_ref, windows, plans, search_budget):
    return dict(schema=SCHEMA, schema_version=1, activity_time=reference,
        windows=windows_ref, plans=plans_ref, search_budget=search_budget,
        window_summary=windows['summary'], summary=plans['summary'],
        parent_graph_modified=False, source_order_relaxation_scope='LOCAL_CONDITIONAL_COMPONENT_MODEL_ONLY',
        source_order_trials_require_independent_lowering_admission=True,
        target_program_executed=False, new_measurements=False, native_compiled=False,
        physical_optimality_proven=False, automatic_transformation_authorized=False,
        deployment_authorized=False)


def prepare_closed_window_plans(activity_time, out, *, search_budget=10000):
    directory = Path(out).resolve()
    if directory.exists():
        raise ValueError('fresh closed-window preparation output required')
    reference = artifact_reference(Path(activity_time).resolve(), TIME_SCHEMA)
    windows, plans, _ = _derive(reference, search_budget)
    directory.mkdir(parents=True, exist_ok=False)
    report = _report(reference, _write(directory/'windows.json', windows),
        _write(directory/'plans.json', plans), windows, plans, search_budget)
    _write(directory/'result.json', report)
    return report


def replay_closed_window_plans(path):
    """Return qualified source context once for downstream C++ trial lowering."""
    reference = artifact_reference(Path(path).resolve(), SCHEMA)
    report = read_artifact(reference)
    if (report.get('schema'), report.get('schema_version')) != (SCHEMA, 1):
        raise ValueError('versioned closed-window preparation required')
    windows, plans, context = _derive(report['activity_time'], report['search_budget'])
    expected = _report(report['activity_time'], report['windows'], report['plans'],
        windows, plans, report['search_budget'])
    if (report != expected or semantic_sha256(read_artifact(report['windows'])) != semantic_sha256(windows)
            or semantic_sha256(read_artifact(report['plans'])) != semantic_sha256(plans)):
        raise ValueError('closed-window plans differ from complete source/time/resource replay')
    read_artifact(reference)
    return report, windows, plans, context


def read_closed_window_plans(path):
    return replay_closed_window_plans(path)[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--activity-time'); source.add_argument('--check')
    parser.add_argument('--out'); parser.add_argument('--search-budget', type=int, default=10000)
    args = parser.parse_args(argv)
    if args.check:
        if args.out:
            parser.error('check consumes only a saved closed-window receipt')
        report = read_closed_window_plans(args.check)
    else:
        if not args.out:
            parser.error('window planning requires --out')
        report = prepare_closed_window_plans(args.activity_time, args.out, search_budget=args.search_budget)
    print(json.dumps(dict(windows=report['window_summary'], plans=report['summary'],
        deployment_authorized=False), sort_keys=True)); return 0


if __name__ == '__main__':
    raise SystemExit(main())

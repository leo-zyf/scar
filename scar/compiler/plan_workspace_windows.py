"""Plan source-closed same-iteration windows without unrolling their parent."""
import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.workspace-closed-window-preparation'


def _derive(reference, search_budget):
    from scar.compiler.refine_workspace_execution import replay_workspace_execution
    from scar.analysis.workspace_closed_windows import derive_workspace_closed_windows
    from scar.optimization.workspace_window_models import plan_workspace_window_models
    report, execution, context = replay_workspace_execution(reference['uri'])
    if report != read_artifact(reference):
        raise ValueError('original workspace execution publication changed during replay')
    refs = (dict(role='execution', artifact=reference, locator='#'), *context['source_refs'])
    windows = derive_workspace_closed_windows(context['schedule'], context['resources'],
        context['workspace_plan'], context['step_plan'], context['native_metadata'], execution,
        source_refs=refs)
    plans = plan_workspace_window_models(context['schedule'], context['resources'], context['costs'],
        context['accounting'], windows, search_budget=search_budget)
    read_artifact(reference)
    return windows, plans, execution


def _implementations():
    root = Path(__file__).resolve().parents[1]
    return {name:artifact_reference(root/name, 'scar.python-source') for name in
        ('compiler/plan_workspace_windows.py', 'analysis/workspace_closed_windows.py',
         'analysis/closed_windows.py', 'optimization/workspace_window_models.py',
         'optimization/closed_window_planner.py', 'optimization/resource_solver.py')}


def _report(reference, windows_ref, plans_ref, windows, plans, execution, search_budget):
    if plans['summary']['selected_transforms'] != 0:
        raise ValueError('window proposals need a separately qualified execution refinement before lowering')
    return dict(schema=SCHEMA, schema_version=1, refinement=reference,
        windows=windows_ref, plans=plans_ref, search_budget=search_budget,
        window_summary=windows['summary'], summary=plans['summary'],
        implementations=_implementations(),
        compiler_handoff=dict(decision='NO_NEW_TRANSFORMATION', backend='NO_OP',
            retained_source_blueprint=execution['native_blueprint']['cpp_source'],
            parent_native_manifest=execution['native_blueprint']['manifest'],
            parent_execution_plan_sha256=execution['native_blueprint']['execution_plan_sha256'],
            reason='NO_COMPLETE_LEGALITY_AND_COST_QUALIFIED_TRANSFORM_SELECTED',
            recompilation_required=False, new_workers_or_buffers_enabled=False,
            original_python_remains_default=True),
        iterations_unrolled=0, root_slot_lifetimes_unchanged=True,
        parent_graph_modified=False, full_numeric_timeline_planned=False,
        target_program_executed=False, new_measurements=False, native_compiled=False,
        physical_optimality_proven=False, default_backend='NO_OP', deployment_authorized=False)


def _write(path, document):
    with path.open('x') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, document['schema'])


def prepare_workspace_window_plans(refinement, out, *, search_budget=10000):
    reference = artifact_reference(Path(refinement).resolve(), 'scar.workspace-execution-refinement-publication')
    directory = Path(out).resolve()
    if directory.exists():
        raise ValueError('fresh workspace window output required')
    windows, plans, execution = _derive(reference, search_budget)
    directory.mkdir(parents=True)
    window_ref = _write(directory/'windows.json', windows)
    plans_ref = _write(directory/'plans.json', plans)
    report = _report(reference, window_ref, plans_ref, windows, plans, execution, search_budget)
    _write(directory/'result.json', report)
    return report


def replay_workspace_window_plans(path):
    """Admit saved local models by cold-replaying their exact original graph."""
    reference = artifact_reference(Path(path).resolve(), SCHEMA)
    report = read_artifact(reference)
    if report.get('schema_version') != 1:
        raise ValueError('versioned workspace window preparation required')
    windows, plans, execution = _derive(report['refinement'], report['search_budget'])
    expected = _report(report['refinement'], report['windows'], report['plans'],
        windows, plans, execution, report['search_budget'])
    if (report != expected or semantic_sha256(read_artifact(report['windows'])) != semantic_sha256(windows)
            or semantic_sha256(read_artifact(report['plans'])) != semantic_sha256(plans)):
        raise ValueError('workspace windows differ from full source/resource/cost/model replay')
    read_artifact(reference)
    return report, windows, plans


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--refinement'); source.add_argument('--check')
    parser.add_argument('--out'); parser.add_argument('--search-budget', type=int, default=10000)
    args = parser.parse_args(argv)
    if args.check:
        if args.out:
            parser.error('check only reads an existing planning result')
        report = replay_workspace_window_plans(args.check)[0]
    else:
        if not args.out:
            parser.error('window planning requires --out')
        report = prepare_workspace_window_plans(args.refinement, args.out, search_budget=args.search_budget)
    print(json.dumps(dict(windows=report['window_summary'], plans=report['summary'],
        handoff=report['compiler_handoff']['decision'], iterations_unrolled=0,
        physical_optimality_proven=False, deployment_authorized=False), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

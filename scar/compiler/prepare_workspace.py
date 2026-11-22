"""Replay a source loop or an explicit call composition before workspace reuse."""
import argparse
import json
from pathlib import Path

from scar.compiler.stateful_model import document, stateful_digest
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.record_codec import encode
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.workspace-preparation'
LOOP_SCHEMA = 'scar.stateful-loop-compiler-build'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _base(reference, region, step_plan):
    if reference is None:
        return None
    raw = read_artifact(reference)
    directory = Path(reference['uri']).parent
    _need(raw.get('schema') == 'scar.native-stateful-artifact', 'original unmodified stateful base artifact required')
    region_ref = artifact_reference(directory/raw['files']['region'], 'scar.stateful-array-region')
    plan_ref = artifact_reference(directory/raw['files']['plan'], 'scar.stateful-stage-plan')
    _need(semantic_sha256(read_artifact(region_ref)) == semantic_sha256(document(region))
        and read_artifact(plan_ref) == step_plan and raw['region_sha256'] == stateful_digest(region)
        and raw['plan_sha256'] == semantic_sha256(step_plan), 'base artifact source region/plan differ from admitted inputs')
    return reference


def _derive(source_kind, reference, repetitions, base_reference):
    from scar.optimization.stateful_workspace import plan_workspace_reuse
    source = read_artifact(reference)
    proof = loop_plan = recipe = None
    refs = dict(source=reference)
    if source_kind == 'ORIGINAL_STATEFUL_LOOP':
        from scar.analysis.build import BuildSession
        from scar.analysis.source_call_view import SourceFunctionCallProvider
        from scar.compiler.frontend.stateful import read_stateful_loop_proof
        from scar.optimization.stateful_loop_schedule import verify_stateful_loop
        _need(source.get('schema') == LOOP_SCHEMA, 'original source-loop compiler publication required')
        read_artifact(source['program'])
        program = Path(source['program']['uri']).parent
        with BuildSession(program) as session:
            _need(artifact_reference(program/'manifest.json','scar.program-build') == source['program'],
                'original loop must retain the exact program graph manifest')
            provider = session.session.providers[source['provider']]
            _need(type(provider) is SourceFunctionCallProvider, 'original source-call provider required')
            proof = read_stateful_loop_proof(source['original_loop_proof']['uri'], view=session.view, provider=provider)
            loop_plan = read_artifact(source['loop_plan'])
            verify_stateful_loop(proof, Path(loop_plan['target']['uri']).parent, loop_plan)
            region, step_plan = proof.step, loop_plan['step_plan']
            _need(repetitions is None or repetitions == loop_plan['trip_count'], 'workspace count differs from original source-loop plan')
            repetitions = loop_plan['trip_count']
            refs.update(program=source['program'], loop_proof=source['original_loop_proof'], loop_plan=source['loop_plan'],
                region=encode(proof.step_reference), target=loop_plan['target'])
        if base_reference is None:
            base_reference = source.get('step_native_artifact')
        _base(base_reference,region,step_plan)
        if base_reference is not None:
            base = read_artifact(base_reference)
            refs['step_plan'] = artifact_reference(Path(base_reference['uri']).parent/base['files']['plan'], 'scar.stateful-stage-plan')
        else:
            raise ValueError('source-loop preparation requires its original step artifact/plan reference')
    elif source_kind == 'FIXED_INPUT_COMPOSITION':
        from scar.optimization.stateful_recipe import verify_stateful_recipe, unpack_stateful_recipe
        recipe = verify_stateful_recipe(reference['uri'])
        _need(recipe == source, 'original source recipe changed during replay')
        region, step_plan = unpack_stateful_recipe(recipe)
        refs.update(region=recipe['artifacts']['region'], step_plan=recipe['artifacts']['stage_plan'],
            target=recipe['artifacts']['target_snapshot'])
        _base(base_reference,region,step_plan)
    else:
        raise ValueError('unsupported workspace source authority')
    if base_reference is not None:
        refs['base_native'] = base_reference
    source_refs = tuple(dict(role=role, artifact=value, locator='#') for role,value in refs.items())
    plan = plan_workspace_reuse(region,step_plan,repetitions,loop_plan=loop_plan,source_refs=source_refs)
    read_artifact(reference)
    context = dict(loop_proof=proof, loop_plan=loop_plan, base_native_reference=base_reference,
        recipe=recipe, source_refs=source_refs, source_reference=reference, source_kind=source_kind,
        region_reference=refs['region'], step_plan_reference=refs['step_plan'])
    return region, step_plan, plan, context


def _report(kind, reference, plan_reference, plan, context):
    root = Path(__file__).resolve().parents[1]
    return dict(schema=SCHEMA, schema_version=1, source_kind=kind, source=reference,
        trip_count=plan['trip_count'], region=context['region_reference'], step_plan=context['step_plan_reference'],
        workspace_plan=plan_reference, base_native=context['base_native_reference'],
        source_refs=encode(context['source_refs']),
        implementations={name:artifact_reference(root/name,'scar.python-source') for name in
            ('compiler/prepare_workspace.py','optimization/stateful_workspace.py')},
        source_authority=('ORIGINAL_F4_CALLER_LOOP_AND_STEP_REPLAY' if kind == 'ORIGINAL_STATEFUL_LOOP'
            else 'EXPLICIT_FIXED_INPUT_COMPOSITION_NOT_ORIGINAL_CALLER_LOOP_PROOF'),
        iterations_unrolled=False, cost_status='UNKNOWN', selected_for_use=False,
        native_compiled=False, target_program_executed=False, performance_measured=False,
        physical_optimality_proven=False, deployment_authorized=False)


def prepare_workspace(source, out, *, source_kind, trip_count=None, base_native=None):
    directory = Path(out).resolve()
    _need(not directory.exists(), 'fresh workspace preparation output required')
    schema = LOOP_SCHEMA if source_kind == 'ORIGINAL_STATEFUL_LOOP' else 'scar.execution-recipe'
    reference = artifact_reference(Path(source).resolve(),schema)
    base_reference = None
    if base_native is not None:
        path = Path(base_native).resolve()
        base_reference = artifact_reference(path/'artifact.json' if path.is_dir() else path,'scar.native-stateful-artifact')
    _, _, plan, context = _derive(source_kind,reference,trip_count,base_reference)
    directory.mkdir(parents=True)
    plan_path = directory/'workspace-plan.json'
    plan_path.write_text(json.dumps(plan,sort_keys=True,indent=2,allow_nan=False)+'\n')
    report = _report(source_kind,reference,artifact_reference(plan_path,plan['schema']),plan,context)
    (directory/'result.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return report


def replay_workspace_preparation(path):
    reference = artifact_reference(Path(path).resolve(),SCHEMA)
    report = read_artifact(reference)
    _need((report.get('schema'),report.get('schema_version')) == (SCHEMA,1), 'versioned workspace preparation required')
    region,step_plan,plan,context = _derive(report['source_kind'],report['source'],report['trip_count'],report['base_native'])
    expected = _report(report['source_kind'],report['source'],report['workspace_plan'],plan,context)
    _need(report == expected and semantic_sha256(read_artifact(report['workspace_plan'])) == semantic_sha256(plan),
        'workspace graph differs from complete source-loop/step/slot-lifetime replay')
    read_artifact(reference)
    return report,region,step_plan,plan,context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--loop-build'); source.add_argument('--recipe'); source.add_argument('--check')
    parser.add_argument('--trip-count',type=int); parser.add_argument('--base-native'); parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check:
        if args.out or args.base_native or args.trip_count is not None: parser.error('check consumes only a saved preparation')
        report = replay_workspace_preparation(args.check)[0]
    else:
        if not args.out: parser.error('workspace preparation requires --out')
        if args.recipe and args.trip_count is None: parser.error('explicit composition requires --trip-count')
        report = prepare_workspace(args.loop_build or args.recipe,args.out,
            source_kind='ORIGINAL_STATEFUL_LOOP' if args.loop_build else 'FIXED_INPUT_COMPOSITION',
            trip_count=args.trip_count,base_native=args.base_native)
    print(json.dumps(dict(source_authority=report['source_authority'],trip_count=report['trip_count'],
        workspace_plan=report['workspace_plan'],cost_status='UNKNOWN',deployment_authorized=False),sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

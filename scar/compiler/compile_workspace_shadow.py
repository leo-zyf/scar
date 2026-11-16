"""Derive and compile COMMIT-region shadow residency, preserving input scans."""
import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.workspace-shadow-residency-preparation'
COMPILED = 'scar.compiled-workspace-shadow-residency'


def _write(path, document):
    with path.open('x') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    return artifact_reference(path, document['schema'])


def _implementations():
    root = Path(__file__).resolve().parents[1]
    return {name:artifact_reference(root/name, 'scar.python-source') for name in
        ('compiler/compile_workspace_shadow.py', 'compiler/workspace_variant.py',
         'optimization/workspace_shadow_residency.py', 'optimization/workspace_guard_properties.py',
         'codegen/workspace_shadow_residency.py')}


def _derive(reference):
    from scar.compiler.refine_workspace_execution import replay_workspace_execution
    from scar.compiler.stateful_model import stateful_from_dict
    from scar.optimization.workspace_shadow_residency import derive_workspace_shadow_residency
    report, execution, context = replay_workspace_execution(reference['uri'])
    if read_artifact(reference) != report:
        raise ValueError('original workspace execution publication changed')
    native = context['native_metadata']; parent = Path(context['native_manifest']['uri']).parent
    region = stateful_from_dict(read_artifact(artifact_reference(parent/native['files']['region'],
        'scar.stateful-array-region')))
    plan = derive_workspace_shadow_residency(region, context['step_plan'], context['workspace_plan'],
        execution, native, source_refs=(dict(artifact=reference, locator='#/certificate'),))
    context['original_native_reference'] = context['native_manifest']
    read_artifact(reference)
    return plan, context


def _preparation(reference, plan_reference, plan, context):
    return dict(schema=SCHEMA, schema_version=1, refinement=reference, residency_plan=plan_reference,
        original_native_manifest=context['native_manifest'], status=plan['status'],
        implementations=_implementations(), source_and_graph_replayed=True,
        input_finite_scans_reused=False, source_values_are_invariant=False,
        native_compiled=False, target_program_executed=False, performance_measured=False,
        trial_only=True, default_backend='NO_OP', deployment_authorized=False)


def prepare_workspace_shadow(refinement, out):
    reference = artifact_reference(Path(refinement).resolve(), 'scar.workspace-execution-refinement-publication')
    directory = Path(out).resolve()
    if directory.exists():
        raise ValueError('fresh shadow residency output required')
    plan, context = _derive(reference)
    directory.mkdir(parents=True)
    plan_reference = _write(directory/'residency.json', plan)
    report = _preparation(reference, plan_reference, plan, context)
    _write(directory/'result.json', report)
    return report


def replay_workspace_shadow(path):
    reference = artifact_reference(Path(path).resolve(), SCHEMA)
    report = read_artifact(reference); plan, context = _derive(report['refinement'])
    if (semantic_sha256(read_artifact(report['residency_plan'])) != semantic_sha256(plan)
            or report != _preparation(report['refinement'], report['residency_plan'], plan, context)):
        raise ValueError('shadow residency differs from complete source/ROI/provenance replay')
    read_artifact(reference)
    return report, plan, context


def _render(path):
    from scar.codegen.workspace_shadow_residency import render_shadow_residency
    reference = artifact_reference(Path(path).resolve(), SCHEMA)
    report, plan, context = replay_workspace_shadow(reference['uri'])
    if plan['status'] != 'PROVEN_CONDITIONAL':
        raise ValueError('shadow residency has not been proved: '+str(plan.get('rejections', [])))
    original = context['native_metadata']; parent = Path(context['native_manifest']['uri']).parent
    source = read_artifact(plan['native_blueprint']['cpp_source'], json_document=False)
    transformed, mapping = render_shadow_residency(source, original, plan)
    headers = {name:(parent/original['files'][name]).read_bytes()
        for name in ('stateful_abi_v1.h', 'scar_abi_v1.h')}
    from hashlib import sha256
    if any(sha256(raw).hexdigest() != original['sha256'][name] for name,raw in headers.items()):
        raise ValueError('original workspace ABI header changed')
    return reference, report, plan, context, transformed, mapping, headers


def compile_workspace_shadow(preparation, out):
    from scar.compiler.workspace_variant import compile_variant
    return compile_variant(_render(preparation), out, kind='shadow-residency')


def replay_compiled_workspace_shadow(path):
    from scar.compiler.workspace_variant import read_variant
    path = Path(path).resolve(); path = path/'result.json' if path.is_dir() else path
    report = read_artifact(artifact_reference(path, COMPILED))
    return read_variant(path, _render(report['preparation']['uri']), kind='shadow-residency')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--refinement'); source.add_argument('--preparation'); source.add_argument('--check')
    parser.add_argument('--out'); args = parser.parse_args(argv)
    if args.check:
        if args.out:
            parser.error('check only consumes an existing result')
        document = json.loads(Path(args.check).read_bytes())
        report = (replay_compiled_workspace_shadow(args.check)[0] if document.get('schema') == COMPILED
                  else replay_workspace_shadow(args.check)[0])
    else:
        if not args.out:
            parser.error('a fresh output directory is required')
        report = (prepare_workspace_shadow(args.refinement, args.out) if args.refinement
                  else compile_workspace_shadow(args.preparation, args.out))
    print(json.dumps({key:report[key] for key in ('status','residency_plan','native_artifact','compact',
        'default_backend','deployment_authorized') if key in report}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

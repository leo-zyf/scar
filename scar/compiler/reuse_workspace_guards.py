"""Prove and lower first-scan/success-induction for closed workspace loops."""
from hashlib import sha256
import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.workspace-guard-reuse-preparation'
NATIVE = 'scar.native-workspace-guard-reuse'
COMPILED = 'scar.compiled-workspace-guard-reuse'


def _write(path, document):
    with path.open('x') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, document['schema'])


def _implementations():
    root = Path(__file__).resolve().parents[1]
    return {name:artifact_reference(root/name, 'scar.python-source') for name in
        ('compiler/reuse_workspace_guards.py', 'optimization/workspace_guard_properties.py',
         'codegen/workspace_guard_reuse.py', 'compiler/workspace_variant.py')}


def _derive(reference):
    from scar.compiler.refine_workspace_execution import replay_workspace_execution
    from scar.compiler.stateful_model import stateful_from_dict
    from scar.optimization.workspace_guard_properties import derive_workspace_guard_properties
    report, execution, context = replay_workspace_execution(reference['uri'])
    if read_artifact(reference) != report:
        raise ValueError('original workspace execution publication changed')
    native = context['native_metadata']
    parent = Path(context['native_manifest']['uri']).parent
    region_reference = artifact_reference(parent/native['files']['region'], 'scar.stateful-array-region')
    region = stateful_from_dict(read_artifact(region_reference))
    plan = derive_workspace_guard_properties(region, context['step_plan'], context['workspace_plan'],
        execution, native, source_refs=(dict(artifact=reference, locator='#/certificate'),))
    read_artifact(reference)
    return plan, context


def _preparation(reference, property_reference, plan, context):
    return dict(schema=SCHEMA, schema_version=1, refinement=reference, property_plan=property_reference,
        original_native_manifest=context['native_manifest'], status=plan['status'],
        implementations=_implementations(), source_and_graph_replayed=True,
        input_contents_are_invariant=False, native_compiled=False, target_program_executed=False,
        performance_measured=False, trial_only=True, default_backend='NO_OP', deployment_authorized=False)


def prepare_workspace_guards(refinement, out):
    reference = artifact_reference(Path(refinement).resolve(), 'scar.workspace-execution-refinement-publication')
    directory = Path(out).resolve()
    if directory.exists():
        raise ValueError('fresh workspace property output required')
    plan, context = _derive(reference)
    directory.mkdir(parents=True)
    prop = _write(directory/'properties.json', plan)
    report = _preparation(reference, prop, plan, context)
    _write(directory/'result.json', report)
    return report


def replay_workspace_guards(path):
    reference = artifact_reference(Path(path).resolve(), SCHEMA)
    report = read_artifact(reference)
    plan, context = _derive(report['refinement'])
    if (semantic_sha256(read_artifact(report['property_plan'])) != semantic_sha256(plan)
            or report != _preparation(report['refinement'], report['property_plan'], plan, context)):
        raise ValueError('workspace guard plan differs from complete source/property replay')
    read_artifact(reference)
    return report, plan, context


def _render(preparation):
    from scar.codegen.workspace_guard_reuse import render_guard_reuse
    reference = artifact_reference(Path(preparation).resolve(), SCHEMA)
    report, plan, context = replay_workspace_guards(reference['uri'])
    if plan['status'] != 'PROVEN_CONDITIONAL':
        raise ValueError('input finite guard reuse has not been proved: '+str(plan.get('rejections', [])))
    parent = Path(context['native_manifest']['uri']).parent
    original = context['native_metadata']
    source = read_artifact(plan['native_blueprint']['cpp_source'], json_document=False)
    transformed, mapping = render_guard_reuse(source, original, plan)
    headers = {key:(parent/original['files'][key]).read_bytes()
               for key in ('stateful_abi_v1.h', 'scar_abi_v1.h')}
    for key, raw in headers.items():
        if sha256(raw).hexdigest() != original['sha256'][key]:
            raise ValueError('original workspace ABI header changed')
    return reference, report, plan, context, transformed, mapping, headers


def compile_workspace_guards(preparation, out):
    from scar.compiler.workspace_variant import compile_variant
    return compile_variant(_render(preparation), out, kind='finite-guards')


def read_compiled_workspace_guards(path):
    from scar.compiler.workspace_variant import read_variant
    reference = artifact_reference(Path(path).resolve(), COMPILED)
    report = read_artifact(reference)
    return read_variant(path, _render(report['preparation']['uri']), kind='finite-guards')[0]


def main(argv=None, *, compile=False):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--preparation' if compile else '--refinement')
    source.add_argument('--check'); parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check:
        if args.out:
            parser.error('check only reads existing publications')
        report = read_compiled_workspace_guards(args.check) if compile else replay_workspace_guards(args.check)[0]
    else:
        if not args.out:
            parser.error('a fresh output directory is required')
        report = compile_workspace_guards(args.preparation, args.out) if compile else prepare_workspace_guards(args.refinement, args.out)
    print(json.dumps({key:report[key] for key in ('property_plan','status','native_artifact','compact',
        'default_backend','deployment_authorized') if key in report}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

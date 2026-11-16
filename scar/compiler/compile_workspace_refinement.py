"""Rebuild a source-qualified workspace execution certificate as a C++20 trial.

The existing backend and compact exporter remain unchanged. Rebuilding must
preserve the certified source blueprint and controller identities; a model
coordinate is neither a runtime clock nor deployment authority.
"""
import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.refined-workspace-compiler-build'
REFINEMENT = 'scar.workspace-execution-refinement-publication'
CERTIFICATE = 'scar.workspace-execution-refinement'
NATIVE = 'scar.native-stateful-workspace-repeat'
EXPORT = 'scar.stateful-workspace-loop-runtime-export'


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _same(left, right):
    return semantic_sha256(left) == semantic_sha256(right)


def _path(path, filename='result.json'):
    path = Path(path).resolve()
    return path/filename if path.is_dir() else path


def _admit(refinement):
    from scar.compiler.refine_workspace_execution import replay_workspace_execution
    reference = artifact_reference(_path(refinement), REFINEMENT)
    report, certificate, context = replay_workspace_execution(reference['uri'])
    _need(read_artifact(reference) == report, 'workspace refinement changed during source/cost/v4 replay')
    _need(_same(read_artifact(report['certificate']), certificate)
          and certificate.get('schema') == CERTIFICATE and certificate.get('schema_version') == 1,
          'original workspace execution certificate required')
    preparation = read_artifact(context['preparation'])
    original = read_artifact(context['native_manifest'])
    blueprint = certificate['native_blueprint']
    _need(set(blueprint) == {'manifest', 'cpp_source', 'workspace_plan_sha256',
                           'execution_plan_sha256', 'base_ids', 'source_mapping'},
          'complete exact native workspace blueprint required')
    _need(_same(original, context['native_metadata'])
          and original.get('schema') == NATIVE
          and blueprint['manifest'] == context['native_manifest']
          and original['preparation'] == context['preparation'],
          'certificate, preparation and original native manifest differ')
    _need(preparation['source_kind'] in ('ORIGINAL_STATEFUL_LOOP', 'FIXED_INPUT_COMPOSITION')
          and preparation.get('base_native') is not None
          and preparation['base_native'] == original['base_native_reference'],
          'exact source reconstruction requires a preparation with its fixed original base native')
    for key in ('workspace_plan_sha256', 'execution_plan_sha256', 'base_ids', 'source_mapping'):
        _need(_same(blueprint[key], original[key]), 'native blueprint differs: '+key)
    _need(semantic_sha256(context['workspace_plan']) == original['workspace_plan_sha256']
          and _same(read_artifact(preparation['workspace_plan']), context['workspace_plan']),
          'workspace lifetime plan differs from the certified source plan')
    old_directory = Path(context['native_manifest']['uri']).parent
    _need(artifact_reference(old_directory/original['files']['source'], blueprint['cpp_source']['schema'])
          == blueprint['cpp_source'], 'native blueprint must reference the actual original C++ source')
    source = read_artifact(blueprint['cpp_source'], json_document=False)
    _need(original['sha256']['source'] == blueprint['cpp_source']['sha256'],
          'native blueprint source digest differs from the original manifest')
    return reference, report, certificate, context, preparation, original, source


def _match_native(directory, metadata, admitted):
    _, _, certificate, context, _, original, source = admitted
    _need(metadata.get('schema') == NATIVE and metadata.get('schema_version') == 1,
          'existing workspace ABI1 native artifact required')
    for key in ('build_id', 'preparation', 'base_native_reference', 'base_ids',
                'workspace_plan_sha256', 'execution_plan_sha256', 'execution_plan',
                'source_mapping', 'implementation_sha256', 'compiler', 'cpu_threads',
                'cpu_ids', 'memory', 'trip_count'):
        _need(_same(metadata[key], original[key]), 'rebuild changed certified native identity: '+key)
    _need(metadata['preparation'] == context['preparation']
          and metadata['execution_plan_sha256'] == certificate['native_blueprint']['execution_plan_sha256']
          and (directory/metadata['files']['source']).read_bytes() == source,
          'rebuild does not implement the exact certified source and controller')
    _need(set(metadata['files']) == set(original['files']), 'rebuild native artifact footprint differs')
    for key in metadata['files']:
        if key != 'library':
            _need(metadata['sha256'][key] == original['sha256'][key],
                  'rebuild changed a source, plan or ABI-header member: '+key)
    _need(metadata.get('trial_only') is True and metadata.get('default_decision') == 'NO_OP'
          and metadata.get('deployment_authorized') is False,
          'workspace refinement is an unmeasured trial, not deployment approval')


def _compact(directory, metadata, preparation):
    if preparation['source_kind'] != 'ORIGINAL_STATEFUL_LOOP':
        _need(not (directory/'bundle').exists(), 'fixed composition cannot acquire a caller-loop bundle')
        return None
    from scar.runtime.native_stateful_repeat import inspect_bundle
    path = directory/'bundle'/'export-report.json'
    reference = artifact_reference(path, EXPORT)
    export = read_artifact(reference)
    bundle = inspect_bundle(directory/'bundle', expected_admission_sha256=export['expected_admission_sha256'])
    contract, admission = bundle['contract'], bundle['admission']
    native_reference = artifact_reference(directory/'native'/'artifact.json', NATIVE)
    retained = admission['original_repeat_manifest']
    _need(export['bundle'] == str(directory/'bundle')
          and artifact_reference(directory/'bundle'/'admission.json', admission['schema']) == export['admission']
          and retained['path'] == native_reference['uri']
          and retained['sha256'] == native_reference['sha256']
          and retained['bytes'] == native_reference['size_bytes']
          and admission['original_workspace_preparation'] == metadata['preparation']
          and admission['original_workspace_plan'] == preparation['workspace_plan'],
          'compact bundle is not exported from this certified source-loop native artifact')
    _need(contract['workspace']['plan_sha256'] == metadata['workspace_plan_sha256']
          and contract['workspace']['execution_plan_sha256'] == metadata['execution_plan_sha256']
          and contract['native']['base_ids'] == metadata['base_ids']
          and contract['native']['repeat']['sha256'] == metadata['sha256']['library']
          and contract['trial_only'] is True and contract['deployment_authorized'] is False,
          'compact workspace controller, ELF or trial authority differs')
    return dict(export=reference, bundle=str(directory/'bundle'),
        expected_admission_sha256=export['expected_admission_sha256'])


def _publication(directory, admitted, metadata, compact):
    reference, report, certificate, context, preparation, _, _ = admitted
    return dict(schema=SCHEMA, schema_version=1, refinement=reference,
        certificate=report['certificate'], preparation=context['preparation'],
        original_native_manifest=context['native_manifest'],
        source_kind=preparation['source_kind'], source_authority=preparation['source_authority'],
        native_artifact=artifact_reference(directory/'native'/'artifact.json', NATIVE),
        native_blueprint=certificate['native_blueprint'],
        workspace_plan=preparation['workspace_plan'], workspace_plan_sha256=metadata['workspace_plan_sha256'],
        execution_plan_sha256=metadata['execution_plan_sha256'], base_ids=metadata['base_ids'],
        source=artifact_reference(directory/'native'/metadata['files']['source'],
            certificate['native_blueprint']['cpp_source']['schema']),
        library=artifact_reference(directory/'native'/metadata['files']['library'], 'scar.native-library'),
        compact=compact, caller_loop_authority=('ORIGINAL_F4_LOOP_AND_LIVE_GUARDS'
            if compact is not None else 'NONE_FIXED_INPUT_COMPOSITION_NATIVE_ONLY'),
        implementations={'compiler': artifact_reference(Path(__file__).resolve(), 'scar.python-source')},
        implementation_scope='UNCHANGED_WORKSPACE_POOL_LIFETIME_AND_PER_STEP_TRANSACTION_CONTROLLER',
        certified_source_bytes_identical=True, controller_identities_identical=True,
        original_numerical_body_unchanged=True, numerical_backend_added=False,
        joint_numerical_planning=False, full_model_timeline_implemented=False,
        model_coordinates_are_runtime_timestamps=False, native_compiled=True,
        target_program_executed=False, performance_measured=False, cost_status='UNKNOWN',
        trial_only=True, default_backend='NO_OP', selected_for_use=False,
        deployment_authorized=False, physical_optimality_proven=False, whole_program_cpp=False)


def compile_workspace_refinement(refinement, out):
    from scar.codegen.stateful_workspace import build_workspace_repeat
    admitted = _admit(refinement)
    directory = Path(out).resolve()
    _need(not directory.exists(), 'fresh refined workspace output required')
    directory.mkdir(parents=True)
    artifact = build_workspace_repeat(admitted[3]['preparation']['uri'], directory/'native')
    metadata = artifact.metadata
    _match_native(directory/'native', metadata, admitted)
    if admitted[4]['source_kind'] == 'ORIGINAL_STATEFUL_LOOP':
        from scar.compiler.export_workspace_loop import export_workspace_loop
        export_workspace_loop(directory/'native', directory/'bundle')
    compact = _compact(directory, metadata, admitted[4])
    read_artifact(admitted[0])
    report = _publication(directory, admitted, metadata, compact)
    (directory/'result.json').write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n')
    return report


def replay_compiled_workspace_refinement(path):
    """Recheck source, cost, v4 and native bytes without compilation or CDLL."""
    from scar.codegen.stateful_workspace import inspect_workspace_repeat
    reference = artifact_reference(_path(path), SCHEMA)
    report = read_artifact(reference)
    _need(report.get('schema_version') == 1, 'versioned refined workspace publication required')
    directory = Path(reference['uri']).parent
    admitted = _admit(report['refinement']['uri'])
    _need(admitted[0] == report['refinement'], 'original refinement reference differs')
    metadata, _ = inspect_workspace_repeat(directory/'native')
    _match_native(directory/'native', metadata, admitted)
    compact = _compact(directory, metadata, admitted[4])
    expected = _publication(directory, admitted, metadata, compact)
    _need(_same(report, expected), 'refined workspace publication differs from full current source replay')
    read_artifact(reference)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--refinement'); source.add_argument('--check')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check:
        if args.out:
            parser.error('check consumes only a saved compiler publication')
        report = replay_compiled_workspace_refinement(args.check)
    else:
        if not args.out:
            parser.error('compilation requires --out')
        report = compile_workspace_refinement(args.refinement, args.out)
    print(json.dumps({key: report[key] for key in ('native_artifact', 'compact',
        'source_kind', 'caller_loop_authority', 'default_backend', 'deployment_authorized')}, sort_keys=True))
    return 0


__all__ = ['compile_workspace_refinement', 'replay_compiled_workspace_refinement', 'main']

if __name__ == '__main__':
    raise SystemExit(main())

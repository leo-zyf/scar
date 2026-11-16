"""Prove and lower first-scan/success-induction for closed workspace loops."""
from hashlib import sha256
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

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
         'codegen/workspace_guard_reuse.py')}


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


def _native_metadata(admitted, directory):
    reference, report, plan, context, source, mapping, headers = admitted
    files = dict(source='workspace-guards.cpp', library='workspace_guards_'+
        mapping['execution_plan_sha256'][:20]+'.so', **{name:name for name in headers})
    return dict(schema=NATIVE, schema_version=1, preparation=reference,
        property_plan=report['property_plan'], original_native_manifest=context['native_manifest'],
        execution_plan=mapping['execution_plan'], execution_plan_sha256=mapping['execution_plan_sha256'],
        property_plan_sha256=semantic_sha256(plan), source_mapping=mapping,
        source_sha256=sha256(source).hexdigest(), compiler=context['native_metadata']['compiler'],
        implementations=_implementations(), files=files,
        sha256={key:sha256((directory/name).read_bytes()).hexdigest() for key, name in files.items()},
        cpp_standard=20, stateful_abi_version=1, repeat_abi_version=1,
        source_math_unchanged=True, transactional_boundaries_unchanged=True,
        input_finite_scans_reused=True, all_other_per_step_checks_retained=True,
        cost_status='UNKNOWN', trial_only=True, default_backend='NO_OP', deployment_authorized=False)


def _inspect_native(directory, admitted):
    reference = artifact_reference(directory/'artifact.json', NATIVE)
    metadata = read_artifact(reference)
    if metadata != _native_metadata(admitted, directory):
        raise ValueError('finite-reuse native metadata differs from full source/property replay')
    if (directory/metadata['files']['source']).read_bytes() != admitted[4]:
        raise ValueError('finite-reuse C++ source differs from exact guarded transformation')
    for name, raw in admitted[6].items():
        if (directory/name).read_bytes() != raw:
            raise ValueError('finite-reuse ABI header differs')
    return metadata


def _export(directory, metadata, admitted):
    from scar.compiler.export_workspace_loop import export_workspace_loop
    from scar.runtime.native_stateful_repeat import (GUARD_REUSE_PROFILE, GUARD_REUSE_FAMILY,
        SCHEMA as RUNTIME, ADMISSION, validate_contract, implementation_sha256, inspect_bundle)
    _, preparation, plan, context, _, mapping, _ = admitted
    original_preparation = read_artifact(context['preparation'])
    if original_preparation['source_kind'] != 'ORIGINAL_STATEFUL_LOOP':
        return None
    output = directory/'bundle'
    # Reuse the complete existing F4 caller export, then replace only the
    # source-qualified repeat implementation. Staging files are temporary.
    with tempfile.TemporaryDirectory(prefix='scar-guard-export-') as temporary:
        basis = Path(temporary)/'basis'
        export_workspace_loop(Path(context['native_manifest']['uri']).parent, basis)
        contract = json.loads((basis/'runtime.json').read_bytes())
        admission = json.loads((basis/'admission.json').read_bytes())
        base_bytes = (basis/contract['native']['base']['library']).read_bytes()
    contract['profile'] = GUARD_REUSE_PROFILE
    contract['execution_family'] = GUARD_REUSE_FAMILY
    contract['workspace']['execution_plan_sha256'] = mapping['execution_plan_sha256']
    library = directory/'native'/metadata['files']['library']
    contract['native']['repeat'] = dict(library=library.name,
        sha256=sha256(library.read_bytes()).hexdigest(), bytes=library.stat().st_size)
    contract['input_finite_scans_reused'] = True
    contract['guard_reuse'] = dict(property_plan_sha256=semantic_sha256(plan),
        parent_execution_plan_sha256=plan['native_blueprint']['execution_plan_sha256'],
        input_array_indices=[row['array_index'] for row in plan['input_scans']],
        policy='FIRST_INPUT_SCAN_THEN_SUCCESS_FINITE_PROPERTY_INDUCTION')
    validate_contract(contract)
    output.mkdir()
    runtime = _write(output/'runtime.json', contract)
    admission.update(profile=GUARD_REUSE_PROFILE, native=contract['native'],
        runtime_sha256=runtime['sha256'], runtime_implementation_sha256=implementation_sha256(),
        guard_reuse_native_manifest=artifact_reference(directory/'native/artifact.json', NATIVE),
        guard_reuse_property_plan=preparation['property_plan'],
        finite_property_replayed=True,
        producer=dict(name='SCAR', exporter_sha256=sha256(Path(__file__).read_bytes()).hexdigest()))
    pin = _write(output/'admission.json', admission)
    (output/contract['native']['base']['library']).write_bytes(base_bytes)
    shutil.copyfile(library, output/library.name)
    inspect_bundle(output, expected_admission_sha256=pin['sha256'])
    export = dict(schema='scar.workspace-guard-reuse-runtime-export', schema_version=1,
        bundle=str(output), profile=GUARD_REUSE_PROFILE, runtime=runtime, admission=pin,
        expected_admission_sha256=pin['sha256'], original_f4_loop_and_step_replayed=True,
        finite_property_replayed=True, mathematical_body_unchanged=True,
        original_repeat_native_manifest=context['native_manifest'],
        guard_reuse_native_manifest=admission['guard_reuse_native_manifest'],
        property_plan=preparation['property_plan'], default_backend='NO_OP', deployment_authorized=False)
    _write(output/'export-report.json', export)
    return export


def _compiled_report(directory, admitted, metadata, compact):
    return dict(schema=COMPILED, schema_version=1, preparation=admitted[0],
        native_artifact=artifact_reference(directory/'native/artifact.json', NATIVE),
        property_plan=admitted[1]['property_plan'],
        execution_plan_sha256=metadata['execution_plan_sha256'],
        compact=compact, cpp_standard=20, original_math_and_transaction_order_retained=True,
        native_compiled=True, native_libraries_loaded=False, target_program_executed=False,
        performance_measured=False, trial_only=True, default_backend='NO_OP', deployment_authorized=False)


def compile_workspace_guards(preparation, out):
    admitted = _render(preparation)
    directory = Path(out).resolve()
    if directory.exists():
        raise ValueError('fresh finite-reuse native output required')
    native = directory/'native'
    native.mkdir(parents=True)
    (native/'workspace-guards.cpp').write_bytes(admitted[4])
    for name, raw in admitted[6].items():
        (native/name).write_bytes(raw)
    compiler = admitted[3]['native_metadata']['compiler']
    if subprocess.run([*compiler['argv'], '--version'], check=True, capture_output=True,
            text=True).stdout.splitlines()[0] != compiler['identity']:
        raise ValueError('original workspace compiler identity changed')
    library = native/('workspace_guards_'+admitted[5]['execution_plan_sha256'][:20]+'.so')
    process = subprocess.run([*compiler['argv'], *compiler['options'], str(native/'workspace-guards.cpp'),
        '-o', str(library), '-lm'], capture_output=True, text=True)
    (native/'compiler-stdout.txt').write_text(process.stdout)
    (native/'compiler-stderr.txt').write_text(process.stderr)
    if process.returncode:
        raise RuntimeError('finite-reuse C++ compilation failed: '+process.stderr)
    metadata = _native_metadata(admitted, native)
    _write(native/'artifact.json', metadata)
    compact = _export(directory, metadata, admitted)
    report = _compiled_report(directory, admitted, metadata, compact)
    _write(directory/'result.json', report)
    return report


def read_compiled_workspace_guards(path):
    reference = artifact_reference(Path(path).resolve(), COMPILED)
    report = read_artifact(reference)
    admitted = _render(report['preparation']['uri'])
    directory = Path(path).resolve().parent
    metadata = _inspect_native(directory/'native', admitted)
    compact = report['compact']
    if read_artifact(admitted[3]['preparation'])['source_kind'] == 'ORIGINAL_STATEFUL_LOOP':
        from scar.runtime.native_stateful_repeat import inspect_bundle, GUARD_REUSE_PROFILE
        export = json.loads((directory/'bundle/export-report.json').read_bytes())
        checked = inspect_bundle(directory/'bundle', expected_admission_sha256=export['expected_admission_sha256'])
        contract = checked['contract']
        if (compact != export or contract['profile'] != GUARD_REUSE_PROFILE
                or contract['workspace']['execution_plan_sha256'] != metadata['execution_plan_sha256']
                or contract['guard_reuse']['property_plan_sha256'] != metadata['property_plan_sha256']
                or contract['native']['repeat']['sha256'] != metadata['sha256']['library']
                or export['guard_reuse_native_manifest'] != report['native_artifact']
                or export['property_plan'] != report['property_plan']):
            raise ValueError('finite-reuse compact source/controller/property bindings differ')
    elif compact is not None or (directory/'bundle').exists():
        raise ValueError('fixed composition cannot export a caller replacement')
    if report != _compiled_report(directory, admitted, metadata, compact):
        raise ValueError('finite-reuse compiled publication differs from full replay')
    read_artifact(reference)
    return report


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

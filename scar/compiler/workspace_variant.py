"""Shared build/export admission for source-qualified workspace variants."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256


def _spec(kind):
    if kind == 'finite-guards':
        return dict(native='scar.native-workspace-guard-reuse', compiled='scar.compiled-workspace-guard-reuse',
            export='scar.workspace-guard-reuse-runtime-export', plan_key='property_plan',
            source='workspace-guards.cpp', prefix='workspace_guards_', hash_key='property_plan_sha256',
            proof_name='finite_property_replayed', evidence_prefix='guard_reuse')
    if kind == 'shadow-residency':
        return dict(native='scar.native-workspace-shadow-residency', compiled='scar.compiled-workspace-shadow-residency',
            export='scar.workspace-shadow-residency-runtime-export', plan_key='residency_plan',
            source='workspace-residency.cpp', prefix='workspace_residency_', hash_key='residency_plan_sha256',
            proof_name='shadow_residency_replayed', evidence_prefix='shadow_residency')
    raise ValueError('unknown workspace source variant')


def _write(path, document):
    with path.open('x') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    return artifact_reference(path, document['schema'])


def _metadata(admitted, native, kind):
    reference, report, plan, context, source, mapping, headers = admitted
    spec = _spec(kind)
    files = dict(source=spec['source'], library=spec['prefix']+mapping['execution_plan_sha256'][:20]+'.so',
        **{name:name for name in headers})
    return dict(schema=spec['native'], schema_version=1, preparation=reference,
        **{spec['plan_key']:report[spec['plan_key']]},
        original_native_manifest=context['native_manifest'],
        execution_plan=mapping['execution_plan'], execution_plan_sha256=mapping['execution_plan_sha256'],
        **{spec['hash_key']:semantic_sha256(plan)}, source_mapping=mapping,
        source_sha256=sha256(source).hexdigest(), compiler=context['native_metadata']['compiler'],
        implementations={**report['implementations'],
            'compiler/workspace_variant.py':artifact_reference(Path(__file__).resolve(), 'scar.python-source')},
        files=files, sha256={key:sha256((native/name).read_bytes()).hexdigest() for key,name in files.items()},
        cpp_standard=20, stateful_abi_version=1, repeat_abi_version=1,
        source_math_unchanged=True, transactional_boundaries_unchanged=True,
        input_finite_scans_reused=kind == 'finite-guards', shadow_snapshots_reused=kind == 'shadow-residency',
        cost_status='UNKNOWN', trial_only=True, default_backend='NO_OP', deployment_authorized=False)


def inspect_native(native, admitted, *, kind):
    spec = _spec(kind)
    metadata = read_artifact(artifact_reference(native/'artifact.json', spec['native']))
    if metadata != _metadata(admitted, native, kind) or (native/metadata['files']['source']).read_bytes() != admitted[4]:
        raise ValueError('workspace variant differs from full source/plan replay')
    for name, raw in admitted[6].items():
        if (native/name).read_bytes() != raw:
            raise ValueError('workspace variant ABI header differs')
    return metadata


def _patch_contract(contract, plan, mapping, kind):
    from scar.runtime.native_stateful_repeat import (GUARD_REUSE_PROFILE, GUARD_REUSE_FAMILY,
        SHADOW_RESIDENCY_PROFILE, SHADOW_RESIDENCY_FAMILY)
    parent = plan['native_blueprint']['execution_plan_sha256']
    if kind == 'finite-guards':
        contract.update(profile=GUARD_REUSE_PROFILE, execution_family=GUARD_REUSE_FAMILY,
            input_finite_scans_reused=True, guard_reuse=dict(property_plan_sha256=semantic_sha256(plan),
                parent_execution_plan_sha256=parent,
                input_array_indices=[r['array_index'] for r in plan['input_scans']],
                policy='FIRST_INPUT_SCAN_THEN_SUCCESS_FINITE_PROPERTY_INDUCTION'))
    else:
        contract.update(profile=SHADOW_RESIDENCY_PROFILE, execution_family=SHADOW_RESIDENCY_FAMILY,
            shadow_snapshots_reused=True, shadow_residency=dict(residency_plan_sha256=semantic_sha256(plan),
                parent_execution_plan_sha256=parent,
                input_array_indices=[r['array_index'] for r in plan['snapshots'] if r['mode'] != 'KEEP_FULL_SNAPSHOT'],
                policy='FIRST_FULL_SNAPSHOT_THEN_REFRESH_COMMITTED_ROI_COMPLEMENT'))
        contract['external_assumptions'] = list(dict.fromkeys([*contract['external_assumptions'],
            'PROVIDER_BORROWED_PORT_ACCESSES_COMPLETE_ON_RETURN_NO_DELAYED_POSTCOMMIT_WRITERS']))
    contract['workspace']['execution_plan_sha256'] = mapping['execution_plan_sha256']


def _export(directory, metadata, admitted, kind):
    from scar.compiler.export_workspace_loop import export_workspace_loop
    from scar.runtime.native_stateful_repeat import (validate_contract, implementation_sha256,
        inspect_bundle)
    _, report, plan, context, _, mapping, _ = admitted
    if read_artifact(context['preparation'])['source_kind'] != 'ORIGINAL_STATEFUL_LOOP':
        return None
    spec = _spec(kind); output = directory/'bundle'
    with tempfile.TemporaryDirectory(prefix='scar-workspace-export-') as temporary:
        basis = Path(temporary)/'basis'
        export_workspace_loop(Path(context['native_manifest']['uri']).parent, basis)
        contract = json.loads((basis/'runtime.json').read_bytes())
        admission = json.loads((basis/'admission.json').read_bytes())
        base_bytes = (basis/contract['native']['base']['library']).read_bytes()
    _patch_contract(contract, plan, mapping, kind)
    library = directory/'native'/metadata['files']['library']
    contract['native']['repeat'] = dict(library=library.name,
        sha256=sha256(library.read_bytes()).hexdigest(), bytes=library.stat().st_size)
    validate_contract(contract)
    output.mkdir(); runtime = _write(output/'runtime.json', contract)
    native_ref = artifact_reference(directory/'native/artifact.json', spec['native'])
    admission.update(profile=contract['profile'], native=contract['native'], runtime_sha256=runtime['sha256'],
        runtime_implementation_sha256=implementation_sha256(),
        producer=dict(name='SCAR', exporter_sha256=sha256(Path(__file__).read_bytes()).hexdigest()),
        **{spec['evidence_prefix']+'_native_manifest':native_ref,
           spec['evidence_prefix']+'_plan':report[spec['plan_key']], spec['proof_name']:True})
    pin = _write(output/'admission.json', admission)
    (output/contract['native']['base']['library']).write_bytes(base_bytes)
    shutil.copyfile(library, output/library.name)
    inspect_bundle(output, expected_admission_sha256=pin['sha256'])
    export = dict(schema=spec['export'], schema_version=1, bundle=str(output), profile=contract['profile'],
        runtime=runtime, admission=pin, expected_admission_sha256=pin['sha256'],
        original_f4_loop_and_step_replayed=True, **{spec['proof_name']:True}, mathematical_body_unchanged=True,
        original_repeat_native_manifest=context['native_manifest'], native_manifest=native_ref,
        **{spec['plan_key']:report[spec['plan_key']]}, default_backend='NO_OP', deployment_authorized=False)
    _write(output/'export-report.json', export)
    return export


def _report(directory, admitted, metadata, compact, kind):
    spec = _spec(kind)
    return dict(schema=spec['compiled'], schema_version=1, preparation=admitted[0],
        native_artifact=artifact_reference(directory/'native/artifact.json', spec['native']),
        **{spec['plan_key']:admitted[1][spec['plan_key']]}, execution_plan_sha256=metadata['execution_plan_sha256'],
        compact=compact, cpp_standard=20, original_math_and_transaction_order_retained=True,
        native_compiled=True, native_libraries_loaded=False, target_program_executed=False,
        performance_measured=False, trial_only=True, default_backend='NO_OP', deployment_authorized=False)


def compile_variant(admitted, out, *, kind):
    spec = _spec(kind); directory = Path(out).resolve()
    if directory.exists():
        raise ValueError('fresh workspace variant output required')
    native = directory/'native'; native.mkdir(parents=True)
    (native/spec['source']).write_bytes(admitted[4])
    for name, raw in admitted[6].items():
        (native/name).write_bytes(raw)
    compiler = admitted[3]['native_metadata']['compiler']
    if subprocess.run([*compiler['argv'], '--version'], check=True, capture_output=True,
            text=True).stdout.splitlines()[0] != compiler['identity']:
        raise ValueError('original workspace compiler identity changed')
    library = native/(spec['prefix']+admitted[5]['execution_plan_sha256'][:20]+'.so')
    process = subprocess.run([*compiler['argv'], *compiler['options'], str(native/spec['source']),
        '-o', str(library), '-lm'], capture_output=True, text=True)
    (native/'compiler-stdout.txt').write_text(process.stdout); (native/'compiler-stderr.txt').write_text(process.stderr)
    if process.returncode:
        raise RuntimeError('workspace variant C++ compilation failed: '+process.stderr)
    metadata = _metadata(admitted, native, kind); _write(native/'artifact.json', metadata)
    report = _report(directory, admitted, metadata, _export(directory, metadata, admitted, kind), kind)
    _write(directory/'result.json', report)
    return report


def read_variant(path, admitted, *, kind):
    spec = _spec(kind); reference = artifact_reference(Path(path).resolve(), spec['compiled'])
    report = read_artifact(reference); directory = Path(path).resolve().parent
    metadata = inspect_native(directory/'native', admitted, kind=kind); compact = report['compact']
    if read_artifact(admitted[3]['preparation'])['source_kind'] == 'ORIGINAL_STATEFUL_LOOP':
        from scar.runtime.native_stateful_repeat import inspect_bundle
        export = json.loads((directory/'bundle/export-report.json').read_bytes())
        checked = inspect_bundle(directory/'bundle', expected_admission_sha256=export['expected_admission_sha256'])
        contract = checked['contract']
        if (compact != export or contract['workspace']['execution_plan_sha256'] != metadata['execution_plan_sha256']
                or contract['native']['repeat']['sha256'] != metadata['sha256']['library']
                or export['native_manifest'] != report['native_artifact']
                or export[spec['plan_key']] != report[spec['plan_key']]):
            raise ValueError('workspace variant compact source/controller bindings differ')
    elif compact is not None or (directory/'bundle').exists():
        raise ValueError('fixed composition cannot export a caller replacement')
    if report != _report(directory, admitted, metadata, compact, kind):
        raise ValueError('workspace variant publication differs from full replay')
    read_artifact(reference)
    return report, metadata, admitted[3]

"""Compile an explicit dirty-step trial using checked current-call root reads."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.compiled-workspace-borrowed-roots'
CAPACITY_EXPORT = 'scar.session-capacity-export'
PLANNING = 'scar.stateful-snapshot-rewrite-preparation'
NATIVE = 'scar.native-workspace-borrowed-roots'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _write(path, document):
    with path.open('x') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, document['schema'])


def _derive(planning_ref, export_ref):
    from scar.codegen.workspace_borrowed_roots import render_borrowed_roots
    from scar.codegen.workspace_shadow_residency import _source_plan
    from scar.compiler.plan_stateful_snapshot import check
    from scar.compiler.session_capacity import replay
    from scar.runtime.session_scope_bundle import inspect_scope_bundle

    planning, proof = check(planning_ref['uri'])
    _need(planning == read_artifact(planning_ref) and planning['candidate'] is not None,
          'one cold-checked snapshot rewrite candidate required')
    export = read_artifact(export_ref)
    admitted = inspect_scope_bundle(export['bundle'], expected_admission_sha256=export['expected_admission_sha256'])
    capacity, scope, _, parent = replay(export['preparation']['uri'])
    _need(capacity['status'] == 'PROVEN_CONDITIONAL' and scope == admitted['spec']
        and scope['unqualified_kernel_policy'] == 'RETAIN_PRIVATE_CAPACITY_FULL_SNAPSHOT'
        and export.get('implementation_planning') == admitted['admission'].get('implementation_planning')
        and admitted['admission']['preparation'] == export['preparation']
        and parent['admission']['sha256'] == admitted['admission']['session_admission_sha256']
        and admitted['parent']['admission'] == read_artifact(parent['admission']),
        'candidate must use the same source-qualified private arena and original runtime')
    workspace_ref = capacity['capacity_inputs']['parent_native_metadata']
    _need(workspace_ref == planning['original_native'],
          'snapshot planner and private arena refer to different original workspace code')
    workspace = read_artifact(workspace_ref)
    session = read_artifact(parent['native'])
    residency_ref = read_artifact(parent['preparation'])['residency_preparation']
    residency = read_artifact(read_artifact(residency_ref)['residency_plan'])
    original_cpp = read_artifact(residency['native_blueprint']['cpp_source'], json_document=False)
    _, step, capture = _source_plan(original_cpp, workspace, residency)
    source, execution = render_borrowed_roots(read_artifact(parent['source'], json_document=False),
        session, original_cpp, workspace, step, capture, proof,
        dict(planning=planning_ref, proof=planning['proof'], demand=planning['demand']))
    _need(planning['candidate'] == dict(option='BORROW_ENTRY_ROOTS_UNTIL_COMMIT',
        elided_roots=execution['selected_inputs'], explicit_trial_only=True),
        'C++ borrowed roots differ from the exact candidate selection')
    context = dict(original_bundle=export['bundle'], original_export=export,
        scope=scope, original_scope_admission=admitted['admission'], original_runtime=admitted['parent']['contract'],
        original_admission=admitted['parent']['admission'], parent=parent,
        compiler=session['compiler'], execution=execution)
    return source, context


def _documents(source_ref, library_ref, context):
    execution = context['execution']
    execution_id = semantic_sha256(execution)
    native = dict(schema=NATIVE, schema_version=1, execution_plan=execution,
        execution_plan_sha256=execution_id, source=source_ref, library=library_ref,
        compiler=context['compiler'], original_session=context['parent']['native'],
        explicit_trial_only=True, deployment_authorized=False)
    runtime = deepcopy(context['original_runtime'])
    runtime['native']['execution_plan_sha256'] = execution_id
    runtime['native']['session'] = dict(library=Path(library_ref['uri']).name,
        sha256=library_ref['sha256'], bytes=library_ref['size_bytes'])
    runtime['snapshot_rewrite'] = execution
    scope = deepcopy(context['scope'])
    scope['kernel_snapshot_policy'] = 'CERTIFIED_BORROWED_CURRENT_ROOTS_ON_DIRTY_STEP'
    return native, runtime, scope


def _binding(context, native_ref, runtime_ref, runtime, scope_ref, admission_ref):
    admission = {**context['original_admission'], 'runtime_sha256':runtime_ref['sha256'],
        'native':runtime['native'], 'native_manifest':native_ref}
    scope_admission = {**context['original_scope_admission'], 'scope_sha256':scope_ref['sha256'],
        'session_admission_sha256':admission_ref['sha256'], 'snapshot_rewrite':native_ref}
    return admission, scope_admission


def prepare(planning, capacity_export, out):
    from scar.runtime.session_scope_bundle import inspect_scope_bundle

    plan = artifact_reference(Path(planning).resolve(), PLANNING)
    export = artifact_reference(Path(capacity_export).resolve(), CAPACITY_EXPORT)
    directory = Path(out).resolve()
    _need(not directory.exists(), 'fresh borrowed-root C++ output required')
    cpp, context = _derive(plan, export)
    native_dir = directory/'native'
    bundle = directory/'bundle'
    native_dir.mkdir(parents=True)
    bundle.mkdir()
    source = native_dir/'session.cpp'
    source.write_bytes(cpp)
    headers = {}
    for name, reference in context['parent']['headers'].items():
        path = native_dir/name
        path.write_bytes(read_artifact(reference, json_document=False))
        headers[name] = artifact_reference(path, reference['schema'])
    compiler = context['compiler']
    library = native_dir/('session_'+semantic_sha256(context['execution'])[:20]+'.so')
    process = subprocess.run([*compiler['argv'], *compiler['options'], str(source),
        '-o', str(library), '-lm'], capture_output=True, text=True, check=False)
    (native_dir/'compiler-stderr.txt').write_text(process.stderr)
    if process.returncode:
        raise RuntimeError('borrowed-root C++ compilation failed: '+process.stderr)
    source_ref = artifact_reference(source, 'scar.cpp-source')
    library_ref = artifact_reference(library, 'scar.native-library')
    native, runtime, scope = _documents(source_ref, library_ref, context)
    native_ref = _write(native_dir/'artifact.json', native)
    runtime_ref = _write(bundle/'runtime.json', runtime)
    scope_ref = _write(bundle/'scope.json', scope)
    original = Path(context['original_bundle'])
    base_name = runtime['native']['base']['library']
    shutil.copyfile(original/base_name, bundle/base_name)
    shutil.copyfile(library, bundle/library.name)
    admission = {**context['original_admission'], 'runtime_sha256':runtime_ref['sha256'],
        'native':runtime['native'], 'native_manifest':native_ref}
    admission_ref = _write(bundle/'admission.json', admission)
    _, scope_admission = _binding(context, native_ref, runtime_ref, runtime, scope_ref, admission_ref)
    scope_admission_ref = _write(bundle/'scope-admission.json', scope_admission)
    inspect_scope_bundle(bundle, expected_admission_sha256=scope_admission_ref['sha256'])
    return _write(directory/'result.json', dict(schema=SCHEMA, schema_version=1,
        snapshot_planning=plan, original_capacity_export=export,
        bundle=str(bundle), native=native_ref, source=source_ref, library=library_ref,
        runtime=runtime_ref, scope=scope_ref, admission=admission_ref, scope_admission=scope_admission_ref,
        headers=headers,
        deployment_authorized=False, default_backend='NO_OP', explicit_trial_only=True))


def check(result):
    """Recompute C++ and validate the compact package without opening the ELF."""
    from scar.compiler.inspect_workspace_session import _elf
    from scar.runtime.session_scope_bundle import inspect_scope_bundle

    reference = artifact_reference(Path(result).resolve(), SCHEMA)
    saved = read_artifact(reference)
    source, context = _derive(saved['snapshot_planning'], saved['original_capacity_export'])
    native, runtime, scope = _documents(saved['source'], saved['library'], context)
    admission, scope_admission = _binding(context, saved['native'], saved['runtime'], runtime,
        saved['scope'], saved['admission'])
    bundle = Path(saved['bundle'])
    actual = inspect_scope_bundle(bundle, expected_admission_sha256=saved['scope_admission']['sha256'])
    for key, expected in (('native', native), ('runtime', runtime), ('scope', scope),
            ('admission', admission), ('scope_admission', scope_admission)):
        _need(read_artifact(saved[key]) == expected, 'borrowed-root member differs from source replay: '+key)
    original_headers = context['parent']['headers']
    _need(saved['headers'].keys() == original_headers.keys() and all(
        read_artifact(saved['headers'][name], json_document=False) == read_artifact(ref, json_document=False)
        for name, ref in original_headers.items()), 'borrowed-root ABI headers differ from the original session')
    _need(read_artifact(saved['source'], json_document=False) == source
        and (bundle/runtime['native']['session']['library']).read_bytes() ==
            read_artifact(saved['library'], json_document=False)
        and actual['spec'] == scope and actual['admission'] == scope_admission
        and saved['deployment_authorized'] is False,
        'borrowed-root C++ or compact package bytes differ')
    raw = _elf(saved['library'], tuple(runtime['native']['symbols'].values()))
    _need(native['execution_plan_sha256'].encode() in raw, 'borrowed-root ELF plan constant differs')
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planning')
    parser.add_argument('--capacity-export')
    parser.add_argument('--out')
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check:
        checked = check(args.check)
        result = dict(cold_replayed=True, bundle=checked['bundle'], native_loaded=False, deployment_authorized=False)
    elif args.planning and args.capacity_export and args.out:
        result = prepare(args.planning, args.capacity_export, args.out)
    else:
        parser.error('--planning, --capacity-export and --out are required')
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Package a verified CPU catalog candidate for explicit native review."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen.aten_provider import _dependencies
from scar.codegen.native_cpu_elf import cpu_executable_dynamic
from scar.codegen.native_provider_bundle import _content
from scar.compiler.cpu_entry_replay import _OUTPUT, _TIMEOUT
from scar.compiler.entry_delivery import _environment_digest
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.cpu-catalog-runtime-package'
_SELECTION = 'scar.whole-entry-main-chain'
_FAMILY = 'scar.cpu-array-task-source-provider'
_RUNTIME = 'scar.native-seeded-array-task-entry'
_MANIFEST = 'artifact.json'
_ENV = 'launch-environment.json'
_ORACLE = 'publication-oracle.json'
_GUARD = 'guard/cpu_capacity_exec'
_TOOLS = ('/usr/bin/taskset', '/usr/bin/timeout')


def _need(ok, reason):
    if not ok:
        raise ValueError('CPU catalog runtime package: ' + reason)


def _relative(name):
    path = PurePosixPath(name)
    _need(not path.is_absolute() and bool(path.parts) and '..' not in path.parts and str(path) == name,
          'expected a canonical package-relative member')
    return path


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    path.chmod(0o400)


def _signature(reference):
    return {key: reference[key] for key in ('sha256', 'size_bytes')}


def _member_reference(root, name, files, schema):
    return dict(uri=str(root/_relative(name)), schema=schema, **files[name])


def _environment(root, files):
    reference = _member_reference(root, _ENV, files, 'scar.cpu-frozen-launch-environment')
    return json.loads(read_artifact(reference, json_document=False))


def _copy(root, relative, reference, *, executable=False, expected_source=None):
    source = Path(reference['uri'])
    if expected_source is not None:
        _need(source == expected_source and source.resolve(strict=True) == source,
              'C++ project member changed location or became a symlink')
    path = root / _relative(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    observed = _content(source, copy_to=path)
    _need(observed == _signature(reference), 'source member differs from its receipt: ' + relative)
    path.chmod(0o500 if executable else 0o400)
    return observed


def _host():
    _need(os.uname().sysname == 'Linux', 'Linux host required')
    return dict(machine=os.uname().machine, installation_sha256=sha256(Path('/etc/machine-id').read_bytes()).hexdigest())


def _loader(path, project, files, cwd, environment):
    dynamic = cpu_executable_dynamic(path)
    listing = subprocess.check_output(['/usr/bin/ldd', str(path)], cwd=cwd, env=environment, text=True)
    loaded = _dependencies(listing)
    _need(set(dynamic['needed']) <= loaded.keys(), 'loader does not resolve the ELF direct dependencies')
    result = {}
    for name, resolved in sorted(loaded.items()):
        _need(not name.startswith(('libpython', 'libtorch_python')), 'ELF requires a Python runtime')
        if resolved is None:
            _need(name == 'linux-vdso.so.1', 'unidentified dependency without a file')
            result[name] = dict(kind='KERNEL_VDSO')
            continue
        library = Path(resolved)
        signature = _content(library)
        if library.is_relative_to(project):
            relative = 'project/' + library.relative_to(project).as_posix()
            _need(relative == 'project/provider/objects/' + name and files.get(relative) == signature,
                  'package-local dependency differs from the declared provider object')
            result[name] = dict(kind='PACKAGE_MEMBER', path=relative, **signature)
        else:
            result[name] = dict(kind='HOST_FILE', path=resolved, **signature)
    return dict(dynamic=dynamic, dependencies=result)


def _admit(selection_ref, staging):
    from scar.compiler.cpu_catalog_entry_trial import emit_cpu_catalog_entry

    selected = read_artifact(selection_ref)
    _need(selection_ref['schema'] == _SELECTION and selected['schema_version'] == 1 and not selected['plan_only'],
          'completed entry selection receipt required')
    deliveries = [row for key in ('trial', 'delivery') if (row := selected[key]) is not None
                  and row.get('provider_replay') is not None]
    _need(len(deliveries) == 1, 'exactly one checked CPU catalog trial required')
    trial, = deliveries
    domain = read_artifact(selected['domain'])
    discoveries = read_artifact(selected['variant_discovery'])
    scope, option_id, saved = trial['scope'], trial['selected_id'], trial['provider_replay']
    options = [row for row in discoveries['options'] if row['id'] == option_id]
    _need(len(options) == 1 and scope == domain['scope']
          and selected['selection']['scope_sha256'] == semantic_sha256(scope)
          and any(row['id'] == option_id and row['legal_status'] != 'INELIGIBLE' for row in domain['options'])
          and trial['final_action'] == 'KEEP_ORIGINAL' and trial['launch_authority'] is None
          and trial['deployment_authorized'] is False, 'candidate does not belong to the retained trial scope')
    option, = options
    plan = option['proof']['emission_plan']
    _need(plan == trial['selected_emission'] and plan['provider_family'] == _FAMILY,
          'candidate does not use the supported standalone argv and report provider')
    capacity = trial.get('capacity_contract') or saved.get('capacity_contract')
    replay = emit_cpu_catalog_entry(option, scope, staging/'not-emitted', plan_only=False, verify_entry=True,
                                    retained_pair_ref=saved['reference'], capacity_contract=capacity)
    review = replay.get('provider_replay')
    _need(review is not None and replay['native']['executable'] == trial['native']['executable']
          and replay['native_compiled'] is False and replay['target_executed'] is False
          and all(review.get(key) == saved.get(key) for key in
                  ('reference', 'compilation', 'provider_compilation', 'scope', 'selection', 'capacity_contract')),
          'selected pair or C++ launch could not cold replay: ' + replay['reason'])
    return scope, option_id, review, plan


def _assemble(root, selection_ref, scope, option_id, review, plan):
    compiled = read_artifact(review['provider_compilation'])
    runtime_ref = compiled['runtime']
    runtime = read_artifact(runtime_ref)
    spec = review['scope']
    candidate = spec['candidate']
    project = Path(runtime_ref['uri']).parent
    binary = runtime['binary']
    _need(runtime_ref['schema'] == _RUNTIME and runtime['cxx_standard'] == 20
          and runtime['python_runtime_required'] is False and runtime['files'][binary] == candidate['executable']
          and runtime['cpu_ids'] == spec['cpu_ids'] and {'CMakeLists.txt', 'src/main.cpp'} <= runtime['files'].keys(),
          'selected C++20 project does not contain the verified executable and sources')
    files = {f'project/{name}': _copy(root, f'project/{name}', reference, executable=name == binary,
                expected_source=project/_relative(name)) for name, reference in runtime['files'].items()}
    pair = read_artifact(review['reference'])
    oracle = pair['invocation']['json_report']['oracle']
    argv = candidate['argv']
    slot = candidate['output_slot']
    leaf = argv[slot].removeprefix(_OUTPUT + '/')
    _need(leaf in pair['invocation']['files'] and list(pair['invocation']['files']) == [leaf]
          and Path(leaf).name == leaf and slot == len(argv) - 1 and argv[slot] == _OUTPUT + '/' + leaf
          and sum(_OUTPUT in argument for argument in argv) == 1, 'exact terminal publication slot required')
    files[_ORACLE] = _copy(root, _ORACLE, oracle)
    environment, cwd = candidate['environment'], spec['working_directory']
    _write_json(root/_ENV, environment)
    files[_ENV] = _content(root/_ENV)
    original_linkage = _loader(project/binary, project, files, cwd, environment)
    _need(original_linkage['dynamic'] == runtime['linkage']['dynamic'], 'native ELF dynamic tags differ from build')
    for name, reference in runtime['linkage'].get('qualified_runtime_dependencies', {}).items():
        row = original_linkage['dependencies'][name]
        _need(row['kind'] == 'HOST_FILE' and row['path'] == reference['uri'] and _signature(row) == _signature(reference),
              'runtime dependency differs from the provider contract: ' + name)
    guard = None
    if capacity := review.get('capacity_replay'):
        retained = capacity['guard']
        guard_source = _GUARD + '.cpp'
        files[_GUARD] = _copy(root, _GUARD, retained['executable'], executable=True)
        files[guard_source] = _copy(root, guard_source, retained['source'])
        guarded = read_artifact(capacity['requested'])
        guard = dict(executable=_GUARD, source=guard_source, compiler=retained['compiler'],
            limit_bytes=guarded['candidate_address_space_limit_bytes'],
            linkage=_loader(Path(retained['executable']['uri']), project, files, cwd, environment))
    return dict(schema=SCHEMA, schema_version=1, policy='SCOPED_CANDIDATE_ONLY', deployment_authorized=False,
        launch_authority=None, decision='KEEP_ORIGINAL', cxx_standard=20, python_runtime_required=False,
        scope_sha256=semantic_sha256(scope), option_id=option_id, host=_host(),
        provenance=dict(selection=selection_ref, entry=plan['entry'], target=scope['target'], provider=plan['authority'],
                        compilation=review['compilation'], pair=review['reference'], capacity=review.get('capacity_replay'),
                        runtime=runtime_ref),
        launch=dict(executable=f'project/{binary}', argv=argv[1:], output_slot=slot - 1, output_leaf=leaf,
                    cpu_ids=spec['cpu_ids'], cwd=cwd, environment=_ENV,
                    environment_sha256=_environment_digest(environment), tools={name: _content(Path(name)) for name in _TOOLS},
                    stdin='DEVNULL', timeout_seconds=_TIMEOUT, guard=guard),
        oracle=dict(file=_ORACLE, runtime_fields=spec['normalized_runtime_fields']),
        linkage=original_linkage, linkage_scope='LOADER_RESOLVED_DT_NEEDED_ON_ADMITTED_HOST', files=files)


def verify_cpu_catalog_runtime(reference):
    """Verify a sealed local package independently of the planning archive."""
    package = read_artifact(reference)
    path = Path(reference['uri'])
    root = path.parent
    _need(reference['schema'] == SCHEMA and path.name == _MANIFEST and path.resolve(strict=True) == path
          and package['schema_version'] == 1 and package['policy'] == 'SCOPED_CANDIDATE_ONLY'
          and package['decision'] == 'KEEP_ORIGINAL' and package['launch_authority'] is None
          and package['deployment_authorized'] is False and package['host'] == _host()
          and package['cxx_standard'] == 20 and package['python_runtime_required'] is False,
          'unsupported package, host or runtime authority')
    files = package['files']
    found = set()
    for member in root.rglob('*'):
        mode = member.lstat().st_mode
        if stat.S_ISREG(mode):
            found.add(member.relative_to(root).as_posix())
        else:
            _need(stat.S_ISDIR(mode), 'package requires ordinary local files and directories')
    _need(found == {*files, _MANIFEST}, 'package member set differs from the manifest')
    for name, expected in files.items():
        _need(_content(root/_relative(name)) == expected, 'package member changed: ' + name)
    launch = package['launch']
    executable = root/_relative(launch['executable'])
    guard = launch['guard']
    executables = [executable] if guard is None else [executable, root/_relative(guard['executable'])]
    _need(all(os.access(file, os.X_OK) and not file.stat().st_mode & (stat.S_ISUID | stat.S_ISGID)
              for file in executables), 'package executables need ordinary native execution permissions')
    environment = _environment(root, files)
    cwd = Path(launch['cwd'])
    slot = launch['output_slot']
    leaf = launch['output_leaf']
    argv = launch['argv']
    cpus = launch['cpu_ids']
    _need(cwd.is_absolute() and cwd.resolve(strict=True) == cwd and cwd.is_dir()
          and launch['environment'] == _ENV and launch['environment_sha256'] == _environment_digest(environment)
          and launch['tools'] == {name: _content(Path(name)) for name in _TOOLS}
          and type(slot) is int and slot == len(argv) - 1 and slot > 0 and Path(leaf).name == leaf
          and leaf not in ('', '.', '..') and argv[slot] == _OUTPUT + '/' + leaf
          and sum(_OUTPUT in argument for argument in argv) == 1
          and launch['stdin'] == 'DEVNULL' and launch['timeout_seconds'] == _TIMEOUT
          and cpus and cpus == sorted(set(cpus)) and set(cpus) <= os.sched_getaffinity(0),
          'frozen Linux launch context or publication slot is unavailable')
    _need(package['linkage'] == _loader(executable, root/'project', files, str(cwd), environment),
          'loader resolution of packaged executable differs from its original')
    if guard is not None:
        _need(guard['executable'] == _GUARD and guard['source'] == _GUARD + '.cpp'
              and type(guard['limit_bytes']) is int and 0 < guard['limit_bytes'] < 1 << 63
              and guard['linkage'] == _loader(executables[-1], root/'project', files, str(cwd), environment),
              'C++20 capacity guard differs from its original loader closure or limit')
    return package


def export_cpu_catalog_runtime(selection_ref, out):
    """Copy the checked C++20 project and seal its explicit runtime boundary."""
    out = Path(out).absolute()
    _need(out.resolve() == out and not out.exists() and not out.is_symlink(), 'fresh canonical package path required')
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-cpu-catalog-', dir=out.parent) as directory:
        root = Path(directory)/'package'
        root.mkdir(mode=0o700)
        scope, option, review, plan = _admit(selection_ref, Path(directory))
        _write_json(root/_MANIFEST, _assemble(root, selection_ref, scope, option, review, plan))
        verify_cpu_catalog_runtime(artifact_reference(root/_MANIFEST, SCHEMA))
        _need(not out.exists() and not out.is_symlink(), 'package destination was occupied during export')
        root.rename(out)
    return artifact_reference(out/_MANIFEST, SCHEMA)


def prepare_cpu_catalog_launch(reference, fresh_root):
    """Return a fully checked invocation for an explicit candidate-only run."""
    package = verify_cpu_catalog_runtime(reference)
    root = Path(reference['uri']).parent
    launch = package['launch']
    run = Path(fresh_root).absolute()
    cwd = Path(launch['cwd'])
    _need(run.resolve() == run and not run.exists() and not run.is_symlink()
          and not run.is_relative_to(root) and not run.is_relative_to(cwd),
          'fresh publication directory must be outside the package and original cwd')
    publication = str(run/'published'/launch['output_leaf'])
    argv = list(launch['argv'])
    argv[launch['output_slot']] = publication
    cpus = ','.join(map(str, launch['cpu_ids']))
    taskset, timeout = _TOOLS
    execution = dict(mode='candidate', cwd=str(cwd), environment=_environment(root, package['files']),
        argv=[taskset, '-c', cpus, timeout, '-k', '2', str(launch['timeout_seconds']),
              str(root/launch['executable']), *argv])
    guard = launch['guard']
    return dict(execution=execution, run_directory=str(run), expected_publication=publication,
        address_space_limit_bytes=None if guard is None else guard['limit_bytes'],
        native_capacity_guard_executable=None if guard is None else str(root/guard['executable']),
        oracle=_member_reference(root, _ORACLE, package['files'], 'scar.cpu-publication-oracle'),
        runtime_fields=package['oracle']['runtime_fields'], deployment_authorized=False)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar export-cpu-catalog-runtime', description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--selection', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--manifest-sha256')
    args = parser.parse_args(argv)
    if args.selection is not None and args.out is not None and args.manifest_sha256 is None:
        result = export_cpu_catalog_runtime(artifact_reference(args.selection.resolve(strict=True), _SELECTION), args.out)
    elif args.check is not None and args.out is None and args.manifest_sha256:
        result = artifact_reference(args.check.absolute(), SCHEMA)
        _need(result['sha256'] == args.manifest_sha256, 'manifest differs from the supplied digest')
        verify_cpu_catalog_runtime(result)
    else:
        parser.error('export with --selection and --out, or verify with --check and --manifest-sha256')
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

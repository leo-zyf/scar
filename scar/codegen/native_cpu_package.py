"""Publish and replay the detached runtime package of a CPU native project."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen.native_project import read_native_manifest
from scar.codegen.native_provider_bundle import (
    PACKAGE_RUNPATH, _content, _dependencies, _elf, _name, cpu_executable_dynamic, relocate_cpu_executable,
)
from scar.ir.v2.schemas import canonical_json
from scar.runtime.aten_provider_identity import qualify_cpu_dependencies


SCHEMA = 'scar.native-cpu-package'
_MANIFEST = 'artifact.json'
_FIELDS = {'schema', 'schema_version', 'executable', 'cpu_linkage', 'runtime_search_path', 'providers', 'files'}


def _unique_members(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError('native CPU package manifest contains a duplicate key')
    return result


def _package_files(root):
    files = set()
    for path in root.rglob('*'):
        relative = path.relative_to(root)
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            if relative.parts[0] not in ('bin', 'lib', 'data', 'licenses'):
                raise ValueError(f'unexpected native CPU package directory: {relative}')
        elif stat.S_ISREG(mode):
            files.add(relative.as_posix())
        else:
            raise ValueError(f'native CPU package member is not a local regular file or directory: {relative}')
    return files


def _check_roles(manifest):
    executable = manifest['executable']
    if type(executable) is not str or '/' not in executable:
        raise ValueError('native CPU package executable must be under bin/')
    prefix, name = executable.split('/', 1)
    if prefix != 'bin':
        raise ValueError('native CPU package executable must be under bin/')
    _name(name)
    providers = manifest['providers']
    if type(providers) is not dict or any(relative != 'lib/' + _name(soname) for soname, relative in providers.items()):
        raise ValueError('native CPU package providers must use their exact SONAME under lib/')
    files = manifest['files']
    required = {executable, 'launch.sh', *providers.values()}
    if (type(files) is not dict or not required <= files.keys()
            or any(not name.startswith(('data/', 'licenses/')) for name in files.keys() - required)):
        raise ValueError('native CPU package file roles differ from the runtime layout')


def _check_dynamic_paths(root, executable, providers):
    dynamic = cpu_executable_dynamic(root / executable)
    if dynamic['rpath'] is not None or dynamic['runpath'] != PACKAGE_RUNPATH:
        raise ValueError('native CPU package executable does not use its package-relative RUNPATH')
    roles = list(dynamic['needed'])
    for soname, relative in providers.items():
        dynamic = _elf(root / relative)
        if (dynamic['soname'] != soname or any(dynamic[tag] not in (None, '$ORIGIN')
                                               for tag in ('rpath', 'runpath'))):
            raise ValueError(f'native CPU package provider has a different SONAME or runtime search path: {relative}')
        roles.extend(dynamic['needed'])
    return {_name(name) for name in roles}


def verify_native_cpu_package(directory, manifest_sha256):
    """Replay detached package integrity and loader resolution without running its entry."""
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError('native CPU package root must be a local directory')
    root = directory.resolve(strict=True)
    if not root.is_dir():
        raise ValueError('native CPU package root must be a local directory')
    found = _package_files(root)
    if _MANIFEST not in found:
        raise ValueError('native CPU package has no local artifact manifest')
    manifest = read_native_manifest(root / _MANIFEST, manifest_sha256, object_pairs_hook=_unique_members,
                                    mismatch='native CPU package manifest differs from its declared digest')
    if (type(manifest) is not dict or manifest.keys() - {'source_project'} != _FIELDS
            or manifest['schema'] != SCHEMA or type(manifest['schema_version']) is not int
            or manifest['schema_version'] != 1 or manifest['runtime_search_path'] != PACKAGE_RUNPATH
            or type(manifest.get('source_project', {})) is not dict):
        raise ValueError('unsupported native CPU package manifest')

    _check_roles(manifest)
    files = manifest['files']
    if found != {*files, _MANIFEST}:
        raise ValueError('native CPU package files differ from its declared members')
    for relative, lock in files.items():
        if canonical_json(_content(root / relative)) != canonical_json(lock):
            raise ValueError(f'native CPU package member differs: {relative}')

    executable, providers = manifest['executable'], manifest['providers']
    for name in (executable, 'launch.sh'):
        path = root / name
        if not os.access(path, os.X_OK) or path.stat().st_mode & (stat.S_ISUID | stat.S_ISGID):
            raise ValueError('native CPU package entry and launcher must be ordinary executable files')
    needed = _check_dynamic_paths(root, executable, providers)
    current = _dependencies(subprocess.check_output(['/usr/bin/ldd', str(root / executable)], env={}, text=True))
    linkage = qualify_cpu_dependencies(current)
    if canonical_json(manifest['cpu_linkage']) != canonical_json(linkage):
        raise ValueError('native CPU package linkage differs from the replayed ELF dependency closure')
    torch_libraries = {row['soname'] for row in linkage['approved_dependencies'] if row['category'] == 'LIBTORCH_CPU'}
    if (not torch_libraries <= providers.keys()
            or not needed <= {PurePosixPath(name).name for name in current}):
        raise ValueError('native CPU package does not provide its admitted LibTorch and ELF dependencies')
    for soname, relative in providers.items():
        if current.get(soname) != str(root / relative):
            raise ValueError(f'native CPU package provider did not resolve within the package: {soname}')
    host = []
    for soname in sorted(current.keys() - providers.keys()):
        resolved = current[soname]
        if resolved is not None and Path(resolved).is_relative_to(root):
            raise ValueError(f'unlisted native CPU package provider resolved within the package: {soname}')
        host.append(PurePosixPath(soname).name)
    return root, dict(schema='scar.native-cpu-package-validation', schema_version=1,
                      manifest_sha256=manifest_sha256, cpu_linkage=linkage, providers=providers,
                      host_dependencies=host, runtime_search_path=PACKAGE_RUNPATH,
                      package_local_linkage_verified=True, source_replayed=False,
                      relocation_scope='PACKAGE_RELATIVE_PROVIDER_RESOLUTION_ON_CURRENT_LINUX_HOST')


def _launch_and_data(project, manifest):
    if manifest['schema'] == 'scar.aten-recurrence-delivery':
        from scar.compiler.compile_aten_recurrence import _render_recurrence_launch

        return _render_recurrence_launch(manifest['cpus']), {}
    from scar.compiler.compile_exported_aten import _render_launch

    profiles = manifest.get('profiles', [None])
    masks, weights = {}, {}
    for label in profiles:
        plan = 'plan' if label is None else f'plan/profiles/{label}'
        execution = json.loads((project / plan / 'execution.json').read_bytes())
        if not masks:
            inputs, outputs = execution['inputs'], execution['output_shapes']
        masks[label] = execution['cpu_ids']
        for row in execution['inputs']:
            if row['kind'] != 'WEIGHT':
                continue
            name = row['path']
            location = Path(name)
            if location.parent != Path('data') or location.as_posix() != name:
                raise ValueError('native weight requires its qualified package-local data path')
            receipt = {key: row[key] for key in ('sha256', 'size_bytes')}
            if receipt != manifest['files'].get(name):
                raise ValueError('native weight differs from its published file ABI')
            weights[name] = receipt
    return _render_launch(inputs, len(outputs), masks), weights


def _copy_member(source, target, expected=None):
    target.parent.mkdir(exist_ok=True)
    found = _content(source, copy_to=target)
    if expected is not None and expected != found:
        raise ValueError(f'packaged file differs from its source receipt: {source.name}')
    return found


def export_native_cpu(project, manifest_sha256, *, out):
    """Keep the verified launch ABI and snapshot the CPU provider shared libraries."""
    from scar.runtime.validate_native_project import verify_native_project

    project, proof = verify_native_project(project, manifest_sha256)
    if 'cpu_linkage' not in proof:
        raise ValueError('build the source project with --require-cpu-only before packaging')
    manifest = read_native_manifest(project / _MANIFEST, manifest_sha256,
                                    mismatch='native project manifest changed after source verification')
    dependencies = manifest['dependencies']
    provider_root = Path(dependencies['libtorch_cpu.so']).parent
    providers = {name: f'lib/{name}' for name, path in dependencies.items()
                 if path is not None and Path(path).parent == provider_root}
    locks = json.loads((project / 'plan/dependencies.json').read_bytes())
    provider_locks = {row['path']: {key: row[key] for key in ('sha256', 'size_bytes')}
                      for row in (locks['libraries'] if type(locks) is dict else locks)}
    launcher, weights = _launch_and_data(project, manifest)
    if (project / 'launch.sh').read_bytes() != launcher.encode():
        raise ValueError('published launcher differs from its admitted input, output and CPU binding')
    if not {'libtorch_cpu.so', 'libc10.so'} <= providers.keys():
        raise ValueError('the native project needs co-installed CPU provider libraries')
    out = Path(out).resolve()
    if out.exists() or out.is_relative_to(project):
        raise ValueError('a fresh output outside the source native project is required')
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-native-cpu-', dir=out.parent) as temporary:
        stage = Path(temporary)
        files = {}
        for name, relative in providers.items():
            source = dependencies[name]
            if Path(relative).parent != Path('lib'):
                raise ValueError('expected a flat native provider filename')
            files[relative] = _copy_member(Path(source), stage / relative, provider_locks.get(source))
        for relative, receipt in weights.items():
            files[relative] = _copy_member(project / relative, stage / relative, receipt)
        license_paths = sorted(provider_root.parent.parent.glob('torch-*.dist-info/licenses/LICENSE'))
        if len(license_paths) == 1:
            relative = 'licenses/LibTorch.txt'
            files[relative] = _copy_member(license_paths[0], stage / relative)
        executable = manifest['executable']
        source_elf = project / executable
        (stage / 'bin').mkdir()
        relocate_cpu_executable(source_elf, stage / executable, original_runpath=str(provider_root))
        original = manifest['files'][executable]
        if _content(source_elf) != original:
            raise ValueError('native executable changed during relocation')
        files[executable] = _content(stage / executable)
        entry = stage / 'launch.sh'
        entry.write_bytes(launcher.encode())
        entry.chmod(0o755)
        files['launch.sh'] = _content(entry)
        package = dict(schema=SCHEMA, schema_version=1, executable=executable,
                       source_project=dict(schema=manifest['schema'], manifest_sha256=manifest_sha256,
                                           executable=original),
                       cpu_linkage=proof['cpu_linkage'], providers=providers, files=files,
                       runtime_search_path=PACKAGE_RUNPATH)
        record = stage / _MANIFEST
        record.write_text(json.dumps(package, sort_keys=True, indent=2, allow_nan=False) + '\n')
        digest = _content(record)['sha256']
        verify_native_cpu_package(stage, digest)
        stage.rename(out)
    return dict(path=str(out / _MANIFEST), sha256=digest, providers=providers)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar export-native-cpu', description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--project', help='source CPU-only native project')
    mode.add_argument('--check', help='verify a published runtime directory')
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--out')
    parser.add_argument('--reference', action='append', default=[])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    if args.project:
        if not args.out or arguments or args.reference:
            parser.error('--project requires --out and takes no runtime inputs')
        result = export_native_cpu(args.project, args.manifest_sha256, out=args.out)
    else:
        if args.out or arguments and not args.reference:
            parser.error('--check takes runtime inputs only with --reference')
        root, result = verify_native_cpu_package(args.check, args.manifest_sha256)
        if args.reference:
            from scar.runtime.validate_native_project import validate_native_outputs

            result['bitwise_output_receipts'] = validate_native_outputs(root, arguments, args.reference)
    print(json.dumps(result, sort_keys=True))
    return 0

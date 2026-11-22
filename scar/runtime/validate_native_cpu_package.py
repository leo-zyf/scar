"""Verify an independent CPU LibTorch runtime package on the current Linux host."""

from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess

from scar.codegen.native_cpu_elf import _dependencies
from scar.codegen.native_cpu_elf import PACKAGE_RUNPATH, cpu_executable_dynamic
from scar.codegen.native_provider_bundle import _content, _elf, _name
from scar.ir.v2.schemas import canonical_json
from scar.runtime.native_cpu_linkage import qualify_cpu_dependencies


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
    raw = (root / _MANIFEST).read_bytes()
    if sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError('native CPU package manifest differs from its declared digest')
    manifest = json.loads(raw, object_pairs_hook=_unique_members)
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

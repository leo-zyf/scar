"""Package an ELF's explicitly resolved DT_NEEDED graph without relinking it.

The receipt stops at the named host libraries. It does not capture libraries
opened by executable code, or establish an ABI outside the qualified host.
Validate in the target process before preloading in the recorded order.
"""

import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from scar.codegen.native_cpu_elf import _dynamic_tags, _name
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.native-provider-bundle'
FORMAT = 'ELF_DT_NEEDED_PRELOAD'
_OBJECTS = 'objects'
_MANIFEST = 'bundle.json'
_ABI = ('class', 'data', 'machine')
_RTLD_DI_LINKMAP = 2



def _object(name):
    return f'{_OBJECTS}/{_name(name)}'


def _elf(path):
    output = subprocess.check_output(['readelf', '-h', '-d', '-W', str(path)], text=True,
                                     env={**os.environ, 'LC_ALL': 'C'})
    header = dict(row.strip().split(':', 1) for row in output.splitlines()
                  if row.strip().startswith(('Class:', 'Data:', 'Machine:', 'Type:')))
    if (not header.get('Type', '').strip().startswith('DYN (Shared object file)')
            or not all(header.get(key.title(), '').strip() for key in _ABI)):
        raise ValueError('expected an ELF shared object with ordinary DT_NEEDED dependencies: ' + str(path))
    dynamic = _dynamic_tags(output, shared=True)
    return {**{key: header[key.title()].strip() for key in _ABI}, **dynamic}


def _content(source, *, copy_to=None):
    with source.open('rb') as stream:
        if copy_to is not None:
            with copy_to.open('xb') as destination:
                while block := stream.read(1024 * 1024):
                    destination.write(block)
            stream.seek(0)
        digest = sha256()
        size = 0
        while block := stream.read(1024 * 1024):
            size += len(block)
            digest.update(block)
    result = dict(sha256=digest.hexdigest(), size_bytes=size)
    if copy_to is not None and result != _content(copy_to):
        raise ValueError('ELF source changed while packaging: ' + str(source))
    return result


def _system_names(values):
    values = [_name(value) for value in values]
    if len(set(values)) != len(values):
        raise ValueError('duplicate host system SONAME')
    return sorted(values)


def _graph(root, objects, system_allowlist):
    system = set(system_allowlist)
    used_system, active, visited = set(), set(), set()
    preload = []
    root_elf = objects[root]['elf']

    def walk(relative):
        if relative in active:
            raise ValueError('cyclic ELF provider graph at ' + relative)
        if relative in visited:
            return
        active.add(relative)
        member = objects[relative]['elf']
        if any(member[key] != root_elf[key] for key in _ABI):
            raise ValueError('ELF provider architecture differs from root: ' + relative)
        for name in member['needed']:
            if name in system:
                used_system.add(name)
                continue
            provider = _object(name)
            if provider not in objects or objects[provider]['elf']['soname'] != name:
                raise ValueError('unresolved exact ELF SONAME ' + name + ' needed by ' + relative)
            walk(provider)
        active.remove(relative)
        visited.add(relative)
        if relative != root:
            preload.append(relative)

    walk(root)
    if visited != set(objects):
        raise ValueError('bundle contains objects outside the root provider graph')
    sonames = [row['elf']['soname'] for row in objects.values() if row['elf']['soname'] is not None]
    if len(set(sonames)) != len(sonames) or set(sonames).intersection(system):
        raise ValueError('ambiguous host and/or packaged ELF SONAME')
    return preload, sorted(used_system)


def build_native_provider_bundle(root_ref, out, *, provider_roots, system_sonames):
    """Snapshot one root and providers from explicit directories into a fresh bundle."""
    read_artifact(root_ref, json_document=False)
    original = Path(root_ref['uri']).resolve(strict=True)
    roots = [Path(path).resolve(strict=True) for path in provider_roots]
    if any(not path.is_dir() for path in roots):
        raise ValueError('explicit provider roots must be directories')
    system = _system_names(system_sonames)
    root = _object(original.name)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.native-provider-', dir=out.parent) as staging:
        directory = Path(staging)
        (directory / _OBJECTS).mkdir()
        objects = {}

        def add(source, relative):
            if relative in objects:
                if objects[relative]['origin'] != str(source):
                    raise ValueError('ambiguous source path for ' + relative)
                return
            bundled = directory / relative
            signature = _content(source, copy_to=bundled)
            if relative == root and any(root_ref[key] != signature[key] for key in signature):
                raise ValueError('packaged root differs from qualified ELF receipt')
            elf = _elf(bundled)
            objects[relative] = dict(path=relative, origin=str(source), elf=elf, **signature)
            bundled.chmod(0o444)
            for name in elf['needed']:
                if name in system:
                    continue
                candidates = {(path / name).resolve(strict=True) for path in roots if (path / name).is_file()}
                if len(candidates) != 1:
                    raise ValueError(f'needed ELF SONAME {name} has {len(candidates)} candidates in explicit roots')
                add(candidates.pop(), _object(name))

        add(original, root)
        preload, used_system = _graph(root, objects, system)
        value = dict(schema=SCHEMA, schema_version=1, format=FORMAT, root=root, preload=preload,
                     members=[objects[relative] for relative in sorted(objects)],
                     system_allowlist=system, system_sonames=used_system)
        manifest = directory / _MANIFEST
        manifest.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
        manifest.chmod(0o444)
        if out.exists():
            raise FileExistsError(out)
        directory.rename(out)
    return artifact_reference(out / _MANIFEST, SCHEMA)


def validate_native_provider_bundle(reference):
    """Replay the bundle's bytes and dependency order without opening original paths."""
    value = read_artifact(reference)
    if (value.get('schema'), value.get('schema_version'), value.get('format')) != (SCHEMA, 1, FORMAT):
        raise ValueError('unsupported ELF provider bundle')
    directory = Path(reference['uri']).parent
    object_dir = directory / _OBJECTS
    if object_dir.is_symlink() or not object_dir.is_dir():
        raise ValueError('expected a local regular ELF objects directory')
    objects = {}
    for row in value['members']:
        relative = row['path']
        if relative != _object(Path(relative).name) or relative in objects:
            raise ValueError('invalid or duplicate package-relative object path')
        path = directory / relative
        if (path.is_symlink() or not path.is_file()
                or _content(path) != {key: row[key] for key in ('sha256', 'size_bytes')}
                or _elf(path) != row['elf']):
            raise ValueError('packaged ELF bytes or metadata differ: ' + relative)
        objects[relative] = row
    if {_object(path.name) for path in object_dir.iterdir()} != set(objects) or value['root'] not in objects:
        raise ValueError('bundled root or member set differs')
    system = _system_names(value['system_allowlist'])
    preload, used_system = _graph(value['root'], objects, system)
    if system != value['system_allowlist'] or preload != value['preload'] or used_system != value['system_sonames']:
        raise ValueError('bundle dependency order or host boundary differs')
    return value


class _LinkMap(ctypes.Structure):
    _fields_ = [('address', ctypes.c_void_p), ('name', ctypes.c_char_p)]


def load_native_provider_bundle(reference):
    """Preload a validated closure before other loader activity in a fresh Linux process."""
    value = validate_native_provider_bundle(reference)
    if sys.platform != 'linux' or not hasattr(os, 'RTLD_NOLOAD') or any(
            os.environ.get(var) for var in ('LD_PRELOAD', 'LD_AUDIT')):
        raise ValueError('exact ELF preloading requires a clean Linux glibc loader')
    libc = ctypes.CDLL(None)
    try:
        dlopen, dlclose, dlinfo = libc.dlopen, libc.dlclose, libc.dlinfo
    except AttributeError as error:
        raise ValueError('the ELF preloader requires glibc dlinfo') from error
    dlopen.argtypes, dlopen.restype = (ctypes.c_char_p, ctypes.c_int), ctypes.c_void_p
    dlclose.argtypes, dlclose.restype = (ctypes.c_void_p,), ctypes.c_int
    dlinfo.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.POINTER(_LinkMap)))
    dlinfo.restype = ctypes.c_int
    mode = os.RTLD_NOW | os.RTLD_LOCAL

    def loaded_path(soname):
        handle = dlopen(os.fsencode(soname), mode | os.RTLD_NOLOAD)
        if handle is None:
            return None
        try:
            mapping = ctypes.POINTER(_LinkMap)()
            if (dlinfo(handle, _RTLD_DI_LINKMAP, ctypes.byref(mapping)) != 0
                    or not mapping or not mapping.contents.name):
                raise ValueError('cannot attest loaded ELF SONAME ' + soname)
            return Path(os.fsdecode(mapping.contents.name)).resolve()
        finally:
            dlclose(handle)

    directory = Path(reference['uri']).parent.resolve()
    sonames = {row['path']: row['elf']['soname'] for row in value['members']}
    for name in sonames.values():
        if name is not None and loaded_path(name) is not None:
            raise ValueError('packaged ELF SONAME already loaded: ' + name)

    providers = []
    for relative in value['preload']:
        path = directory / relative
        providers.append(ctypes.CDLL(str(path), mode=mode))
        if loaded_path(sonames[relative]) != path:
            raise ValueError('preloaded ELF SONAME did not resolve to packaged object: ' + relative)
    library = ctypes.CDLL(str(directory / value['root']), mode=mode)
    library._scar_provider_libraries = tuple(providers)
    return library


__all__ = ['SCHEMA', 'build_native_provider_bundle', 'validate_native_provider_bundle', 'load_native_provider_bundle']

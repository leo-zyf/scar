"""Pin the installed NVIDIA host compiler and numerical providers for one entry."""

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import platform
import subprocess

from scar.codegen.native_provider_bundle import _content, _dependencies, _elf
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.aten_provider_identity import file_stat_identity, local_file_identity, verify_library_file


SCHEMA = 'scar.operator-stack'
FAMILY = 'NVIDIA_HPC'
_ROUTES = {'CPU': ('cpu_blas',), 'CUDA': ('cudart', 'cublas'),
           'HYBRID': ('cpu_blas', 'cudart', 'cublas')}
_SYSTEM = {'libc', 'libm', 'libmvec', 'libdl', 'libpthread', 'librt', 'libstdc++',
           'libgcc_s', 'libatomic', 'libutil', 'libresolv'}
_VERIFIED_COMPONENTS = ContextVar('verified_nvidia_stack_components', default=None)


@contextmanager
def verify_stack_components_once():
    """Reuse a verified local file identity during one stack admission or replay."""
    active = _VERIFIED_COMPONENTS.get()
    if active is not None:
        yield
        return
    token = _VERIFIED_COMPONENTS.set({})
    try:
        yield
    finally:
        _VERIFIED_COMPONENTS.reset(token)


def stack_component_content(path):
    """Hash a component once per unchanged local identity in the active replay."""
    active = _VERIFIED_COMPONENTS.get()
    if active is None:
        return _content(path)
    resolved = path.resolve(strict=True)
    with resolved.open('rb') as member:
        status = file_stat_identity(os.fstat(member.fileno()))
        identity = local_file_identity(member)
        previous = active.get(resolved) if identity is not None else None
        if previous is not None and previous[0] == identity:
            content = previous[1]
        else:
            content = verify_library_file(member)
        if (status != file_stat_identity(os.fstat(member.fileno())) or status != file_stat_identity(resolved.stat())
                or path.resolve(strict=True) != resolved):
            raise ValueError('NVIDIA stack component changed during verification: ' + str(path))
        if identity is not None:
            active[resolved] = (identity, content)
    return dict(content)


def _pin(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), **stack_component_content(path))


def _run(command):
    env = {key: value for key, value in os.environ.items() if not key.startswith('LD_')}
    return subprocess.check_output(command, env=dict(env, LC_ALL='C'), text=True, timeout=20)


def _inside(path, root):
    path = path.resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError('operator stack component escapes the selected SDK: ' + str(path))
    return path


def _library(path, root):
    resolved = _inside(path, root)
    soname = _elf(resolved)['soname']
    if not soname or (path.parent / soname).resolve(strict=True) != resolved:
        raise ValueError('operator library lacks a matching runtime SONAME: ' + str(path))
    return dict(_pin(resolved), soname=soname, link_path=str(path.parent / soname))


def _nvidia(root, version):
    root = Path(root).resolve(strict=True)
    arch = platform.machine()
    if root.parent.name != 'Linux_' + arch or arch not in ('x86_64', 'aarch64'):
        raise ValueError('SDK root does not match the current Linux architecture')
    host = root / 'compilers'
    compiler = _inside(host / 'bin/nvc++', root)
    release = _run([str(compiler), '--version']).strip().splitlines()[0]
    if not release.startswith('nvc++ ' + root.name + '-'):
        raise ValueError('nvc++ release differs from the selected SDK root')
    # This initial host ABI has been qualified against the x86-64 SDK's LP64 BLAS.
    possible_blas = host / 'lib/libblas_lp64.so'
    libraries, headers = {}, {}
    if arch == 'x86_64' and possible_blas.is_file():
        libraries['cpu_blas'] = _library(possible_blas, root)
        headers['cpu_blas'] = [_pin(_inside(host/'include'/name, root))
                               for name in ('cblas.h', 'openblas_config.h')]
    if version is None:
        default = root / 'cuda/include'
        version = _inside(default, root).relative_to(root/'cuda').parts[0] if default.exists() else None
    if version is not None:
        if not isinstance(version, str) or len(version.split('.')) != 2 or not all(
                part.isascii() and part.isdigit() for part in version.split('.')):
            raise ValueError('CUDA release must name one installed major.minor SDK variant')
        runtime, math = root/'cuda'/version, root/'math_libs'/version
        libraries.update(cudart=_library(runtime/'lib64/libcudart.so', root),
                         cublas=_library(math/'lib64/libcublas.so', root),
                         cublaslt=_library(math/'lib64/libcublasLt.so', root))
        headers['cuda'] = [_pin(_inside(runtime/'include'/name, root))
                           for name in ('cuda.h', 'cuda_runtime.h', 'cuda_runtime_api.h')]
        headers['cublas'] = [_pin(_inside(math/'include'/name, root))
                             for name in ('cublas_v2.h', 'cublas_api.h')]
    if not libraries:
        raise ValueError('the SDK exposes no supported numerical component')

    runtime = {}
    for row in libraries.values():
        for soname, resolved in _dependencies(_run(['ldd', row['path']])).items():
            if resolved is None:
                continue
            path = Path(resolved).resolve(strict=True)
            if path.is_relative_to(root) and soname not in runtime:
                runtime[soname] = _pin(path)
            if soname in runtime and runtime[soname]['path'] != str(path):
                raise ValueError('one SONAME resolves to multiple installed SDK files: ' + soname)
    return dict(schema=SCHEMA, schema_version=1, family=FAMILY, root=str(root), architecture=arch,
        sdk_release=root.name, cuda_release=version, compiler=dict(_pin(compiler), version=release),
        libraries=libraries, headers=headers, sdk_dependencies=runtime,
        cpu_blas_integer_abi='LP64' if 'cpu_blas' in libraries else None,
        numerical_contract=dict(dtype='FLOAT32', accumulation='FLOAT32',
                                reduced_precision_allowed=False, per_operator_stack_switch_allowed=False))


def publish_nvidia_stack(root, out, *, cuda_version=None):
    document = _nvidia(root, cuda_version)
    with Path(out).open('x') as output:
        json.dump(document, output, indent=2, sort_keys=True)
        output.write('\n')
    return artifact_reference(out, SCHEMA)


def load_operator_stack(value):
    """Rebuild the lock from the installed SDK without loading numerical code."""
    reference = value if isinstance(value, dict) else artifact_reference(Path(value).resolve(), SCHEMA)
    document = read_artifact(reference)
    if (reference['schema'] != SCHEMA or document['family'] != FAMILY or document['schema_version'] != 1
            or _nvidia(document['root'], document['cuda_release']) != document):
        raise ValueError('operator stack does not match the installed NVIDIA SDK')
    read_artifact(reference, json_document=False)
    return reference, document


def nvidia_stack_build(document, route):
    names = _ROUTES[route]
    rows = [document['libraries'].get(name) for name in names]
    if not all(rows):
        raise ValueError('the pinned NVIDIA stack does not provide the requested placement: ' + route)
    include = ('cpu_blas',) if route == 'CPU' else ('cuda', 'cublas')
    if route == 'HYBRID':
        include += ('cpu_blas',)
    libraries = [Path(row['link_path']) for row in rows]
    return dict(compiler=[document['compiler']['path']], libraries=libraries,
        include_dirs=[str(Path(document['headers'][key][0]['path']).parent) for key in include],
        runtime_dirs=list(dict.fromkeys(str(path.parent) for path in libraries)))


def verify_operator_stack_linkage(document, dependencies, route):
    """Prove the recorded ELF closure uses one SDK and exact installed bytes."""
    required = {name: document['libraries'][name] for name in _ROUTES[route]}
    selected = {row['soname']: row for row in required.values()}
    if route != 'CPU':
        lt = document['libraries']['cublaslt']
        selected[lt['soname']] = lt
    if not selected.keys() <= dependencies.keys():
        raise ValueError('ELF is missing a numerical component of its selected operator stack')
    # Only components required by this route qualify. In particular, the
    # accelerator image cannot accidentally pull CPU BLAS from another stack.
    other_numerical = {row['soname'] for row in document['libraries'].values()} - selected.keys()
    records = []
    for name, raw in sorted(dependencies.items()):
        if name == 'linux-vdso.so.1' and raw is None:
            continue
        if raw is None or not Path(raw).is_absolute():
            raise ValueError('unresolved ELF dependency: ' + name)
        path = Path(raw).resolve(strict=True)
        record = _pin(path)
        expected = selected.get(name, document['sdk_dependencies'].get(name))
        if name in other_numerical:
            raise ValueError('numerical library is outside the selected route: ' + name)
        if expected is not None:
            if any(record[key] != expected[key] for key in record):
                raise ValueError('loaded operator library differs from the locked component: ' + name)
        elif (not path.is_relative_to('/usr/lib') and not path.is_relative_to('/usr/lib64')
              or not (name.partition('.so')[0] in _SYSTEM
                      or Path(name).name.startswith(('ld-linux-x86-64.so.', 'ld-linux-aarch64.so.')))):
            raise ValueError('ELF dependency is outside the selected operator stack: ' + name)
        records.append(dict(soname=name, **record))
    return dict(schema='scar.operator-stack-linkage', schema_version=1, family=FAMILY,
                route=route, sdk_release=document['sdk_release'], dependencies=records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--nvhpc-root')
    operation.add_argument('--check')
    parser.add_argument('--cuda-version')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.nvhpc_root:
        if args.out is None:
            parser.error('--out is required when pinning an installation')
        reference = publish_nvidia_stack(args.nvhpc_root, args.out, cuda_version=args.cuda_version)
        document = read_artifact(reference)
    else:
        if args.out is not None or args.cuda_version is not None:
            parser.error('--check consumes the retained lock')
        reference, document = load_operator_stack(args.check)
    print(json.dumps(dict(stack=reference, release=document['sdk_release'],
                          cuda=document['cuda_release'], components=sorted(document['libraries'])), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Qualify the loader-resolved Linux dependencies of a native CPU LibTorch entry."""

from pathlib import PurePosixPath
import re


_LOADER = {'ld-linux-x86-64', 'ld-linux-aarch64'}
_CATEGORIES = {
    'LINUX_LOADER': _LOADER,
    'LINUX_HOST_RUNTIME': {'libc', 'libm', 'libmvec', 'libdl', 'libpthread', 'librt', 'libutil', 'libresolv', 'libz'},
    'HOST_COMPILER_RUNTIME': {'libstdc++', 'libgcc_s', 'libatomic', 'libgfortran', 'libquadmath',
                              'libc++', 'libc++abi', 'libunwind'},
    'LIBTORCH_CPU': {'libtorch_cpu', 'libc10'},
    'HOST_PARALLEL_RUNTIME': {'libgomp', 'libomp', 'libiomp5', 'libtbb', 'libtbbmalloc', 'libtbbmalloc_proxy'},
    'HOST_MATH_PROVIDER': {'libblas', 'libcblas', 'liblapack', 'liblapacke', 'libopenblas', 'libblis', 'libatlas',
                           'libmkl_rt', 'libmkl_core', 'libmkl_sequential', 'libmkl_intel_lp64', 'libmkl_intel_ilp64',
                           'libmkl_gf_lp64', 'libmkl_gf_ilp64', 'libmkl_gnu_thread', 'libmkl_intel_thread',
                           'libmkl_tbb_thread'},
}
_APPROVED = {name: case for case, names in _CATEGORIES.items() for name in names}
_SHARED_OBJECT = re.compile(
    r'([a-zA-Z][a-zA-Z0-9_+-]*?)(?:-(?:[0-9a-f]{8,64}|[0-9]{1,6}(?:\.[0-9]{1,6}){1,3}))?'
    r'\.so(?:\.[0-9]{1,6}){0,4}'
)
_GLIBC_LOADER = re.compile(r'ld-[0-9]{1,6}\.[0-9]{1,6}\.so')
_VDSO = 'linux-vdso.so.1'


def _stem(name):
    match = _SHARED_OBJECT.fullmatch(name)
    return match[1] if match is not None else None


def _absolute_path(value):
    if type(value) is not str:
        return None
    path = PurePosixPath(value)
    if path.root != '/' or '..' in path.parts or path.as_posix() != value:
        return None
    return path


def qualify_cpu_dependencies(dependencies: dict[str, str | None]) -> dict:
    """Issue a deterministic CPU qualification from a complete resolved ELF census."""
    if type(dependencies) is not dict or not dependencies or any(type(name) is not str for name in dependencies):
        raise ValueError('CPU native linkage requires an explicit Linux ELF dependency census')
    approved, rejected, observed = [], [], set()
    for soname, resolved in sorted(dependencies.items()):
        if soname == _VDSO and resolved is None:
            approved.append(dict(soname=soname, category='LINUX_KERNEL_VDSO'))
            continue
        declaration = _absolute_path(soname) if '/' in soname else PurePosixPath(soname)
        stem = None if declaration is None else _stem(declaration.name)
        path = _absolute_path(resolved)
        if (stem not in _APPROVED or path is None
                or (_stem(path.name) != stem and not (stem in _LOADER and _GLIBC_LOADER.fullmatch(path.name)))):
            rejected.append(soname)
            continue
        approved.append(dict(soname=soname, category=_APPROVED[stem]))
        observed.add(stem)
    if rejected:
        raise ValueError('CPU native linkage rejected ELF dependencies: ' + ', '.join(rejected))
    if not {'libtorch_cpu', 'libc10', 'libc'} <= observed or not observed & _LOADER:
        raise ValueError('CPU native linkage requires LibTorch CPU, c10, libc and the Linux loader')
    return dict(schema='scar.cpu-native-linkage', schema_version=1,
                scope='LINUX_ELF_LOADER_RESOLVED_DT_NEEDED_CLOSURE',
                approved_dependencies=approved, platform_portability_proven=False)

"""Aten provider identity."""

from hashlib import sha256
import os
import stat
import sys
import re
from pathlib import Path, PurePosixPath
from uuid import UUID


# Observe file identity for trusted local LibTorch installations.
_LOCAL_FILESYSTEMS = {b'ext4', b'xfs', b'btrfs'}


def file_stat_identity(status):
    """Retain the same fields for descriptor and pathname observations."""
    return dict(device=status.st_dev, inode=status.st_ino, mode=status.st_mode,
                uid=status.st_uid, gid=status.st_gid, size_bytes=status.st_size,
                mtime_ns=status.st_mtime_ns, ctime_ns=status.st_ctime_ns)


def _descriptor_mount(fd):
    with Path(f'/proc/self/fdinfo/{fd}').open('rb') as info:
        mount_id = next((line.partition(b':')[2].strip() for line in info if line.startswith(b'mnt_id:')), b'')
    if not mount_id.isdigit():
        return None
    with Path('/proc/self/mountinfo').open('rb') as mounts:
        for line in mounts:
            prefix, _, suffix = line.partition(b' - ')
            if prefix.partition(b' ')[0] != mount_id:
                continue
            fields = suffix.split(maxsplit=1)
            if fields and fields[0] in _LOCAL_FILESYSTEMS:
                return int(mount_id), fields[0].decode('ascii')
            return None
    return None


def local_file_identity(member):
    """Return a same-boot metadata witness for an open local file, if supported.

    Comparing these witnesses can reuse an observed SHA256 for a trusted
    installation. The witness describes the installed file's lifecycle and
    assumes its storage is not modified outside that lifecycle.
    """
    if sys.platform != 'linux':
        return None
    try:
        fd = member.fileno()
        status = os.fstat(fd)
        if not stat.S_ISREG(status.st_mode):
            return None
        mount = _descriptor_mount(fd)
        if mount is None:
            return None
        boot_id = str(UUID(Path('/proc/sys/kernel/random/boot_id').read_text(encoding='ascii').strip()))
        namespace = Path('/proc/self/ns/mnt').stat()
    except (OSError, ValueError):
        return None
    mount_id, filesystem = mount
    return dict(file_stat_identity(status), boot_id=boot_id,
                mount_namespace=dict(device=namespace.st_dev, inode=namespace.st_ino),
                mount_id=mount_id, filesystem=filesystem)


def verify_library_file(member, receipt=None):
    """Hash the opened library from offset zero and check its retained bytes."""
    member.seek(0)
    digest = sha256()
    size = 0
    while data := member.read(1024 * 1024):
        digest.update(data)
        size += len(data)
    actual = dict(sha256=digest.hexdigest(), size_bytes=size)
    if receipt is not None and (receipt['sha256'], receipt['size_bytes']) != (actual['sha256'], size):
        raise ValueError(f'ATen trial library differs from pinned bytes: {member.name}')
    return actual


# Qualify the loader-resolved Linux dependencies of a native CPU LibTorch entry.
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

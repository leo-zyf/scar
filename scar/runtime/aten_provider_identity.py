"""Observe file identity for trusted local LibTorch installations."""

from hashlib import sha256
import os
from pathlib import Path
import stat
import sys
from uuid import UUID


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

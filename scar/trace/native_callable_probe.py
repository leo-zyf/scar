"""Native callable probe."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import shlex
import struct
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import ModuleType


# Explicit, isolated build/load of SCAR's optional CPython callable probe.
#
# No import compiles code. The returned native addresses describe the current
# process's loaded instance, never an identity across runs or an effect summary.
# Unavailable C fields have no key; a present object field containing None is
# the actual Py_None object and must still pass through the reference recorder.
_MODULE = '_native_callable_probe'


def _check_abi() -> None:
    if (sys.implementation.name != 'cpython' or sys.version_info[:3] != (3, 14, 7)
            or sys.version_info.releaselevel != 'final' or sys.platform != 'linux'
            or platform.machine() != 'x86_64' or struct.calcsize('P') != 8
            or sysconfig.get_config_var('Py_GIL_DISABLED')
            or not sys._is_gil_enabled()):
        raise RuntimeError('native callable probe requires CPython 3.14.7 GIL Linux x86-64')


def build_probe(out_dir: str | Path) -> Path:
    """Compile into a new dedicated directory, preserving source/build receipt.

    Existing directories are rejected. No environment package is installed,
    changed or imported by the compiler invocation. A failed build retains its
    source, command, return code and compiler output for inspection.
    """
    _check_abi()
    output = Path(out_dir).absolute()
    source = Path(__file__).with_name('_native_callable_probe.c').read_bytes()
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    include = sysconfig.get_config_var('INCLUDEPY')
    linker = sysconfig.get_config_var('LDSHARED')
    if not all(type(value) is str and value for value in (suffix, include, linker)):
        raise RuntimeError('CPython extension build configuration is incomplete')
    output.mkdir(parents=True, exist_ok=False)
    copied = output / '_native_callable_probe.c'
    copied.write_bytes(source)
    binary = output / (_MODULE + suffix)
    command = [*shlex.split(linker), *shlex.split(sysconfig.get_config_var('CFLAGS') or ''),
        '-std=c11', '-Wextra', '-Werror=implicit-function-declaration', '-I', include,
        str(copied), '-o', str(binary)]
    receipt = {'schema': 'scar.native-callable-probe-build', 'version': 1,
        'python': sys.version, 'soabi': sysconfig.get_config_var('SOABI'),
        'source_sha256': hashlib.sha256(source).hexdigest(), 'source': copied.name,
        'binary': binary.name, 'command': command, 'address_scope': 'CURRENT_PROCESS_LOADED_INSTANCE'}
    receipt_path = output / 'build.json'
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    receipt.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
    if completed.returncode == 0:
        receipt['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    if completed.returncode:
        raise RuntimeError('native callable probe compilation failed; see ' + str(receipt_path))
    return binary


def load_probe(path: str | Path) -> ModuleType:
    """Load an explicitly chosen binary without compiling or installing it."""
    _check_abi()
    binary = Path(path).absolute()
    if not binary.is_file() or not binary.name.endswith(sysconfig.get_config_var('EXT_SUFFIX')):
        raise ValueError('native callable probe requires an existing current-ABI extension file')
    spec = importlib.util.spec_from_file_location(_MODULE, binary)
    if spec is None or spec.loader is None:
        raise ImportError('cannot create native callable probe extension loader')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Explicit OS observation scopes for program interfaces, not OS internals.
#
# These contracts describe requested selection. Neither an argv nor a successful
# exit establishes loss-free delivery or effective seccomp acceleration.
PROGRAM_INTERFACE_CALLS = (
    # Process/handle identity controls required by the existing continuity reader.
    'clone', 'clone3', 'fork', 'vfork', 'execve', 'execveat', 'exit', 'exit_group',
    'unshare', 'close_range', 'open', 'openat', 'openat2', 'creat', 'close',
    'dup', 'dup2', 'dup3', 'fcntl', 'pipe', 'pipe2', 'socket', 'socketpair',
    # File/range/materialization and program-visible metadata.
    'read', 'write', 'readv', 'writev', 'pread64', 'pwrite64', 'preadv', 'pwritev',
    'preadv2', 'pwritev2', 'lseek', 'stat', 'lstat', 'fstat', 'newfstatat', 'statx',
    'readlink', 'readlinkat', 'getcwd', 'chdir', 'fchdir', 'getdents', 'getdents64',
    'truncate', 'ftruncate', 'fallocate', 'fsync', 'fdatasync', 'sync_file_range',
    'rename', 'renameat', 'renameat2', 'link', 'linkat', 'unlink', 'unlinkat',
    'mkdir', 'mkdirat', 'rmdir', 'copy_file_range', 'sendfile', 'splice', 'tee', 'vmsplice',
    # Allocation/mapping interfaces; not allocator or VM implementation details.
    'brk', 'mmap', 'mremap', 'munmap', 'mprotect', 'madvise',
    'mlock', 'munlock', 'mlockall', 'munlockall',
    # External endpoints, device commands and explicit asynchronous IO interfaces.
    'bind', 'listen', 'connect', 'accept', 'accept4', 'getsockname', 'getpeername',
    'sendto', 'recvfrom', 'sendmsg', 'recvmsg', 'sendmmsg', 'recvmmsg', 'shutdown',
    'setsockopt', 'getsockopt', 'ioctl', 'getrandom',
    'poll', 'ppoll', 'select', 'pselect6', 'epoll_create', 'epoll_create1',
    'epoll_ctl', 'epoll_wait', 'epoll_pwait', 'epoll_pwait2', 'eventfd', 'eventfd2',
    'io_uring_setup', 'io_uring_enter', 'io_uring_register', 'io_setup', 'io_destroy',
    'io_submit', 'io_cancel', 'io_getevents', 'io_pgetevents',
    'wait4', 'waitid', 'kill', 'tkill', 'tgkill',
)


def selection_contract(profile):
    if profile not in ('all', 'program-interfaces'):
        raise ValueError('unknown OS capture selection profile')
    selected = PROGRAM_INTERFACE_CALLS if profile == 'program-interfaces' else ()
    return {
        'profile': profile,
        'expression': 'trace=' + (','.join(selected) if selected else 'all'),
        'explicit_syscalls': list(selected),
        'scope': 'PROGRAM_RESOURCE_INTERFACES' if selected else 'ALL_SYSCALLS',
        'excluded_scope': ['unlisted_syscalls', 'scheduler_and_thread_runtime_internals'] if selected else [],
        'kernel_filter': 'REQUESTED_NOT_VERIFIED' if selected else 'NOT_REQUESTED',
        'event_completeness': 'UNKNOWN',
        'effect_completeness': 'UNKNOWN',
        'absence_inference_authorized': False,
    }


def validate_launch_selection(intent, launch, options):
    """Check the exact declared scope against original actual-launch options.

    Old captures have no new declaration and retain their original checks.
    The caller must still run the existing actual-options/identity-control
    coverage validator; this function never substitutes for it.
    """
    first, second = intent.get('selection_contract'), launch.get('selection_contract')
    if first is None and second is None:
        return None
    if type(first) is not dict:
        raise ValueError('OS selection contract is missing or malformed')
    expected = selection_contract(first.get('profile'))
    if first != expected or second != expected or launch.get('selection_profile') != expected['profile']:
        raise ValueError('intent/launch OS selection contracts differ')
    selected = None
    for index, option in enumerate(options):
        expression = (options[index + 1] if option == '-e' and index + 1 < len(options)
                      else option[2:] if option.startswith('-e') and len(option) > 2 else None)
        if expression is not None and expression.startswith('trace='):
            selected = expression
    if (selected != expected['expression'] or
            ('--seccomp-bpf' in options) != (expected['kernel_filter'] == 'REQUESTED_NOT_VERIFIED')):
        raise ValueError('actual OS selection/filter request differs from its contract')
    return expected


__all__ = [
    'build_probe',
    'load_probe',
    'PROGRAM_INTERFACE_CALLS',
    'selection_contract',
    'validate_launch_selection',
]

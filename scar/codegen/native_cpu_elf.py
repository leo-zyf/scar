"""Inspect CPU runtime ELF dependencies and relocate a fixed RUNPATH in place on a copy."""

from os import fsencode, link
from pathlib import Path
import re
import stat
from subprocess import CalledProcessError, check_output
from tempfile import TemporaryDirectory

_NAME = re.compile(r'[A-Za-z0-9_+][A-Za-z0-9_.+-]*\Z')
PACKAGE_RUNPATH = '$ORIGIN/../lib'
_TAG = re.compile(r'^\s*0x[0-9a-fA-F]+\s+\(([^)]+)\)')
_STRING = re.compile(r'[^\[]*\[(.*)\]\s*\Z')
_EXTRA_LOAD = ('AUDIT', 'DEPAUDIT', 'FILTER', 'AUXILIARY')


_LDD_ENTRY = re.compile(
    r'[ \t]*(?:(?P<soname>[A-Za-z0-9_+.-]+)[ \t]+=>[ \t]+(?P<resolved>/[^\x00\r\n]+)'
    r'|(?P<direct>/[^\x00\r\n]+)|(?P<virtual>linux-(?:vdso(?:64)?|gate)\.so\.1))'
    r'[ \t]\(0x[0-9a-fA-F]+\)[ \t]*'
)


def _name(value):
    if type(value) is not str or not _NAME.fullmatch(value):
        raise ValueError('expected a plain ELF file or SONAME: ' + repr(value))
    return value


def _dependencies(listing):
    result = {}
    for line in listing.splitlines():
        # The loader prints raw paths followed by one separator and the address.
        entry = _LDD_ENTRY.fullmatch(line)
        if entry is None:
            raise ValueError('native ELF dependency was not resolved: ' + line)
        name = entry['soname'] or entry['direct'] or entry['virtual']
        path = entry['resolved'] or entry['direct']
        if name in result:
            raise ValueError('ambiguous ELF dependency: ' + name)
        result[name] = None if path is None else str(Path(path).resolve(strict=True))
    return result


def cpu_executable_dynamic(path) -> dict:
    """Read ordinary ELF lookup tags without running the file."""
    try:
        listing = check_output(['/usr/bin/readelf', '-W', '-d', str(Path(path).absolute())],
                               env={'LC_ALL': 'C'}, encoding='utf-8', errors='surrogateescape')
    except CalledProcessError as error:
        raise ValueError('cannot read ELF dynamic tags') from error
    dynamic = dict(runpath=None, needed=[], rpath=None)
    for row in listing.splitlines():
        tag = _TAG.match(row)
        if tag is None:
            continue
        name = tag[1]
        if name in _EXTRA_LOAD:
            raise ValueError(f'unsupported ELF loader redirection: {name}')
        if name not in ('NEEDED', 'RPATH', 'RUNPATH'):
            continue
        value = _STRING.fullmatch(row[tag.end():])
        if value is None:
            raise ValueError(f'cannot decode ELF lookup tag: {name}')
        if name == 'NEEDED':
            dynamic['needed'].append(_name(value[1]))
        elif dynamic[name.lower()] is None:
            dynamic[name.lower()] = value[1]
        else:
            raise ValueError(f'duplicate ELF lookup tag: {name}')
    if len(set(dynamic['needed'])) != len(dynamic['needed']):
        raise ValueError('duplicate ELF dependency name')
    return dynamic


def relocate_cpu_executable(source, target, *, original_runpath) -> None:
    """Publish a fresh executable with one exact, size-preserving string change."""
    source, target = Path(source), Path(target)
    mode = source.stat().st_mode
    if not stat.S_ISREG(mode) or not mode & 0o111 or mode & (stat.S_ISUID | stat.S_ISGID):
        raise ValueError('an ordinary executable file is required for ELF relocation')
    before = cpu_executable_dynamic(source)
    if (type(original_runpath) is not str or not original_runpath or '\0' in original_runpath
            or before['rpath'] is not None or before['runpath'] != original_runpath):
        raise ValueError('executable does not have the declared single RUNPATH')
    original = fsencode(original_runpath)
    packaged = fsencode(PACKAGE_RUNPATH)
    if len(packaged) > len(original):
        raise ValueError('package RUNPATH does not fit in the original ELF string')

    payload = source.read_bytes()
    marker = b'\0' + original + b'\0'
    start = payload.find(marker)
    if start == -1 or payload.find(marker, start + 1) != -1:
        raise ValueError('original RUNPATH must identify one complete ELF string')
    start += 1
    end = start + len(original)
    expected = payload[:start] + packaged.ljust(len(original), b'\0') + payload[end:]
    with TemporaryDirectory(prefix='.scar-cpu-elf-', dir=target.parent) as temporary:
        staged = Path(temporary) / 'executable'
        staged.write_bytes(expected)
        staged.chmod(stat.S_IMODE(mode))
        after = cpu_executable_dynamic(staged)
        if after != dict(before, runpath=PACKAGE_RUNPATH) or staged.read_bytes() != expected:
            raise ValueError('ELF relocation did not preserve all other executable bytes and lookup tags')
        link(staged, target)

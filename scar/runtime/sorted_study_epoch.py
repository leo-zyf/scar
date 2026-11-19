"""Build a CPython mutation epoch helper for isolated source studies."""

import argparse
import ctypes as ffi
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig
from tempfile import TemporaryDirectory

from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.sorted-study-epoch'
_SOURCE = Path(__file__).resolve().parents[1] / 'native/sorted_study_epoch.cpp'
_BINARY = 'libscar_study_epoch.so'
_COMMANDS = {
    'open': ((), ffi.c_int),
    'watch_dict': ((ffi.py_object,), ffi.c_int),
    'watch_type': ((ffi.py_object,), ffi.c_int),
    'value': ((), ffi.c_uint64),
    'close': ((), ffi.c_int),
}


def _require(conditions, message):
    if not conditions:
        raise ValueError(message)


def _interpreter():
    _require(sys.implementation.name == 'cpython' and sys.version_info >= (3, 12)
             and getattr(sys, '_is_gil_enabled', lambda: True)(), 'mutation epoch needs CPython 3.12+ with a GIL')
    return dict(version=sys.hexversion, soabi=sysconfig.get_config_var('SOABI'))


class ScopeEpoch:
    def __init__(self, path):
        self.library = ffi.PyDLL(str(path))
        self.commands = {}
        for key, (arguments, result) in _COMMANDS.items():
            action = getattr(self.library, f'scar_study_epoch_{key}_v1')
            action.argtypes, action.restype = arguments, result
            self.commands[key] = action
        self.value = self.commands['value']
        self.watch_dict = self.commands['watch_dict']
        self.watch_type = self.commands['watch_type']

    def __enter__(self):
        _require(self.commands['open']() == 0, 'a mutation epoch is already active')
        return self

    def __exit__(self, exc_type, exc, traceback):
        _require(self.commands['close']() == 0, 'mutation epoch cleanup failed')


def _probe(binary):
    class Shape:
        field = 1

    root = {'value': 2}
    with ScopeEpoch(binary) as epoch:
        start = epoch.value()
        _require(start > 0 and epoch.watch_dict(root) == 0 and epoch.watch_type(Shape) == 0,
                 'could not watch exact dictionary and class namespace')
        _require(epoch.value() == start, 'watch registration invalidated the current epoch')
        root['value'] = 3
        root['value'] = 2
        Shape.field = 4
        Shape.field = 1
        _require(epoch.value() == start + 4, 'dictionary or class mutation and restoration were not both observed')
        del root['value']
        _require(epoch.value() == start + 5, 'value removal was not observed')
    _require(epoch.value() == 0, 'mutation watcher was not removed')
    return dict(mutation_and_restore_events=4, delete_events=1, cleanup=True)


def _local(root, path, schema):
    return {**artifact_reference(root / path, schema), 'uri':path}


def build(out):
    interpreter = _interpreter()
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    _require(compiler, 'C++ compiler required')
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-study-epoch-', dir=out.parent) as temporary:
        stage = Path(temporary) / 'package'
        stage.mkdir()
        source = stage / 'epoch.cpp'
        source.write_bytes(_SOURCE.read_bytes())
        binary = stage / _BINARY
        command = [*compiler, '-std=c++20', '-O3', '-fPIC', '-shared', '-fvisibility=hidden',
                   '-Wall', '-Wextra', '-Werror', '-I' + sysconfig.get_path('include'), str(source), '-o', str(binary)]
        subprocess.run(command, check=True)
        receipt = dict(schema=SCHEMA, schema_version=1, interpreter=interpreter,
                       source=_local(stage, 'epoch.cpp', 'scar.cpp-source'),
                       binary=_local(stage, _BINARY, 'scar.elf'), validation=_probe(binary),
                       scope='CPYTHON_STUDY_ADAPTER_ONLY')
        (stage / 'artifact.json').write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
        stage.rename(out)
    return receipt


def load(path, digest, *, verify_probe=False):
    reference = artifact_reference(Path(path).resolve(strict=True), SCHEMA)
    _require(reference['sha256'] == digest, 'mutation epoch helper receipt differs from admitted study')
    record = read_artifact(reference)
    root = Path(reference['uri']).parent
    _require(record['interpreter'] == _interpreter() and record['scope'] == 'CPYTHON_STUDY_ADAPTER_ONLY'
             and record['source'] == _local(root, 'epoch.cpp', 'scar.cpp-source')
             and record['binary'] == _local(root, _BINARY, 'scar.elf')
             and (root / 'epoch.cpp').read_bytes() == _SOURCE.read_bytes(),
             'mutation epoch helper differs from the original study and interpreter')
    binary = root / _BINARY
    if verify_probe:
        _require(_probe(binary) == record['validation'], 'mutation epoch helper did not reproduce guard behavior')
    return reference, ScopeEpoch(binary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--out', type=Path)
    group.add_argument('--check', type=Path)
    args = parser.parse_args(argv)
    if args.out:
        result = build(args.out)
    else:
        reference = artifact_reference(args.check.resolve(strict=True), SCHEMA)
        load(args.check, reference['sha256'], verify_probe=True)
        result = read_artifact(reference)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()

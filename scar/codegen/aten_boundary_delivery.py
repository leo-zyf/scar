"""Compile a replayed standalone ATen trial with an explicit native adapter."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen.aten_provider import _dependencies
from scar.codegen.native_provider_bundle import _content
from scar.compiler.aten_boundary_replay import replay_aten_boundary
from scar.compiler.plan_aten_boundary import _checked, _require, _write


_TEMPLATE = Path(__file__).with_name('aten_boundary_v1.cpp.in')
_HEADER = Path(__file__).with_name('aten_boundary_abi_v1.h')
_ABI = 1
_EXE = 'aten-boundary'


def _array(values):
    _require(type(values) is list and values and all(type(v) is int and 0 <= v < 2**63 for v in values),
             'bounded integer geometry required for the native adapter')
    return ', '.join(map(str, values))


def _literal(value):
    _require(type(value) is str and value.isascii() and '\x00' not in value, 'bounded native literal required')
    return json.dumps(value)


def _library_rows(libraries):
    rows = []
    for library in libraries:
        path = _checked(library)
        stat = path.stat()
        modified = divmod(stat.st_mtime_ns, 10**9)
        changed = divmod(stat.st_ctime_ns, 10**9)
        numbers = (stat.st_dev, stat.st_ino, stat.st_size, *modified, *changed)
        rows.append('    {' + ', '.join((_literal(str(path)), *map(str, numbers))) + '},')
    return '\n'.join(rows)


def _render(context, mode, provider_rows):
    measurement = context['measurement_condition']
    inputs = context['inputs']['contract']
    placement = context['placements'][mode]
    cpu_ids = measurement['availability_mask']
    env = measurement['environment']
    _require(bool(env) and all(key.isascii() and key.isidentifier() and not key.startswith('LD_') for key in env),
             'valid retained launch environment required')
    fields = dict(SHAPE=_array(inputs['shape']), STRIDES=_array(inputs['strides']),
        LAUNCH_CPUS=_array(cpu_ids), MODE={'SERIAL': 'Serial', 'TWO_CPU': 'TwoCpu'}[mode],
        CALLER=str(measurement['caller_cpu']), WORKER=str(measurement['worker_cpu']),
        INTRAOP=str(placement['aten_intraop']), INTEROP=str(placement['aten_interop']),
        DTYPE=_literal(inputs['dtype']), BOUNDARY=_literal(context['boundary']),
        SOURCE_SHA=_literal(measurement['source']['sha256']),
        ENVIRONMENT='\n'.join('    {' + _literal(key) + ', ' + _literal(value) + '},'
                              for key, value in sorted(env.items())),
        PROVIDERS=provider_rows)
    code = _TEMPLATE.read_text()
    for field, value in fields.items():
        placeholder = '@' + field + '@'
        _require(code.count(placeholder) == 1, 'invalid ATen boundary template: ' + field)
        code = code.replace(placeholder, value)
    return code


def _stage_member(stage, name, receipt):
    original = _checked(receipt)
    installed = stage / name
    installed.write_bytes(original.read_bytes())
    _require(_content(installed) == {key: receipt[key] for key in ('sha256', 'size_bytes')},
             'native member changed during staging: ' + name)


def compile_aten_boundary(plan_dir, *, selection_sha256, adapter, adapter_sha256, out):
    """Construct native ELF with one sealed mode and original provider call source."""
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    checked = replay_aten_boundary(plan_dir, expected_selection_sha256=selection_sha256)
    mode = checked['selection']['selected_mode']
    context = checked['evidence']['context']
    measurement = context['measurement_condition']
    linkage = json.loads(_checked(measurement['linkage']).read_bytes())
    adapter = Path(adapter).resolve(strict=True)
    supplied = dict(path=str(adapter), **_content(adapter))
    _require(supplied['sha256'] == adapter_sha256, 'declared adapter SHA256 differs')
    _require(all(0 <= cpu < 1024 for cpu in measurement['availability_mask']),
             'native adapter affinity mask exceeds the supported CPU bitset')
    libraries = context['provider']['libraries']
    provider_rows = _library_rows(libraries)
    command_env = {key: value for key, value in os.environ.items() if not key.startswith('LD_')}
    command_env.update(measurement['environment'])
    # The native launcher fixes the measured environment before the ELF is loaded.
    launch = ['env', '-i', 'PATH=/usr/bin:/bin',
              *(key + '=' + value for key, value in sorted(measurement['environment'].items())),
              'taskset', '-c', ','.join(map(str, measurement['availability_mask'])), './' + _EXE]
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-compile-aten-', dir=out.parent) as tmp:
        stage = Path(tmp)
        for name, receipt in (('measured-source.cpp', measurement['source']),
                              ('operators.hpp', measurement['generated_header']),
                              ('adapter.hpp', supplied)):
            _stage_member(stage, name, receipt)
        (stage / _HEADER.name).write_bytes(_HEADER.read_bytes())
        (stage / 'entry.cpp').write_text(_render(context, mode, provider_rows))
        command = [*linkage['compiler'], *linkage['compile_flags'], '-O2', '-pthread', 'entry.cpp',
                   *linkage['link_flags'], '-o', _EXE]
        subprocess.run(command, cwd=stage, check=True, env=command_env)
        listing = subprocess.run(['ldd', './' + _EXE], cwd=stage, env=command_env,
                                 check=True, capture_output=True, text=True).stdout
        expected = _checked(measurement['dependency_listing']).read_text()
        dependencies = _dependencies(listing)
        _require(dependencies == _dependencies(expected)
                 and not any('libpython' in name or 'libtorch_python' in name for name in dependencies),
                 'native executable dependency closure differs from measured standalone trial')
        _require(provider_rows == _library_rows(libraries), 'provider image changed during compilation')
        script = stage / 'launch.sh'
        script.write_text('#!/bin/sh\nset -eu\ncd -- "$(dirname -- "$0")"\nexec ' + shlex.join(launch) + '\n')
        script.chmod(0o755)
        (stage / 'ldd.txt').write_text(listing)
        files = {p.name: _content(p) for p in sorted(stage.iterdir())}
        receipt = dict(schema='scar.aten-standalone-delivery', schema_version=1, cxx_standard=20,
            adapter_abi_version=_ABI, selection_sha256=selection_sha256, case_id=checked['case_id'],
            plan_receipts=checked['receipts'],
            selected_mode=mode, reserved_cpu_ids=context['placements'][mode]['cpu_ids'],
            launch_cpu_ids=measurement['availability_mask'], input_contract=context['inputs']['contract'],
            original_measurement=context['original_observation'], original_adapter=supplied,
            plan_directory=str(Path(plan_dir).resolve()), compiler_command=command, launch=launch,
            provider=context['provider'], dependencies=dependencies, files=files,
            scope=context['boundary'], cost_remeasured=False, source_replacement_authorized=False)
        _write(stage / 'artifact.json', receipt)
        stage.rename(out)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar compile-aten-boundary', description=__doc__)
    parser.add_argument('--plan', required=True, help='retained plan-aten-boundary output directory')
    parser.add_argument('--selection-sha256', required=True)
    parser.add_argument('--adapter', required=True, help='explicit native adapter exposing the versioned entry')
    parser.add_argument('--adapter-sha256', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    receipt = compile_aten_boundary(args.plan, selection_sha256=args.selection_sha256,
        adapter=args.adapter, adapter_sha256=args.adapter_sha256, out=args.out)
    print(json.dumps(dict(selected_mode=receipt['selected_mode'], executable=str(Path(args.out) / _EXE),
                          launcher=str(Path(args.out) / 'launch.sh'))))
    return 0

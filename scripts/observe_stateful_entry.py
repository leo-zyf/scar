"""Retain two array boundaries and real publication effects of a source entry."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import sys

from scar.ir.execution_recipe import artifact_reference
from scar.trace.code_snapshot import describe_code


SCHEMA = 'scar.stateful-entry-observation'
_CONTEXT = ('HOME', 'LANG', 'LC_ALL', 'MPLBACKEND', 'MPLCONFIGDIR', 'OPENBLAS_NUM_THREADS',
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NPY_DISABLE_CPU_FEATURES', 'NPY_ENABLE_CPU_FEATURES',
            'PYTHONHASHSEED', 'PYTHONPATH')


def _phase(frame, fields, scalars, payload):
    import numpy as np

    state = vars(frame.f_locals['self'])
    arrays = {}
    for field in fields:
        value = state[field]
        if type(value) is not np.ndarray or value.dtype.str != '<f8' or not value.flags.c_contiguous:
            raise ValueError(f'expected contiguous little-endian float64 state for {field}')
        data = value.tobytes()
        arrays[field] = dict(shape=list(value.shape), dtype='float64', byte_order='little',
                             size_bytes=len(data), sha256=sha256(data).hexdigest())
        if payload is not None:
            payload.write(data)
    return dict(code=describe_code(frame.f_code), arrays=arrays, scalars={key: state[key] for key in scalars})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('entry', 'source', 'out', 'effects-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('initial', 'final', 'update', 'dump'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--fields', nargs='+', required=True)
    parser.add_argument('--scalars', nargs='+', required=True)
    parser.add_argument('argv', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ['--'] else args.argv
    entry, source = (getattr(args, name).resolve(strict=True) for name in ('entry', 'source'))
    source_ref, entry_ref = (artifact_reference(path, 'scar.python-source') for path in (source, entry))
    output = args.out.resolve()
    effects = args.effects_dir.resolve()
    if effects.exists():
        parser.error('expected a fresh directory for workload output effects')
    output.mkdir()
    initial, stdout, stderr = (output / name for name in ('initial.bin', 'original.stdout', 'original.stderr'))
    phases = {}
    calls = dict(updates=0, dumps=0)
    milestones = {args.initial: 'initial', args.final: 'final'}
    event_calls = {args.update: 'updates', args.dump: 'dumps'}
    if len(set((*milestones, *event_calls))) != 4 or len(set(args.fields)) != len(args.fields):
        parser.error('expected distinct source roles and array fields')
    positional = [str(entry), *argv]
    original_argv, original_path = sys.argv, list(sys.path)
    sys.argv = positional
    sys.path.insert(0, str(entry.parent))
    source_name = str(source)

    with initial.open('xb') as binary, stdout.open('x', encoding='utf-8') as out, stderr.open('x', encoding='utf-8') as err:
        def observe(frame, event, value):
            code = frame.f_code
            if code.co_filename != source_name:
                return
            target = code.co_qualname
            if event == 'call' and target in event_calls:
                calls[event_calls[target]] += 1
            if event == 'return' and target in milestones:
                phase = milestones[target]
                if phase in phases:
                    raise ValueError(f'ambiguous {phase} boundary')
                phases[phase] = _phase(frame, args.fields, args.scalars, binary if phase == 'initial' else None)

        with redirect_stdout(out), redirect_stderr(err):
            old = sys.getprofile()
            sys.setprofile(observe)
            try:
                runpy.run_path(str(entry), run_name='__main__')
            finally:
                sys.setprofile(old)
                sys.argv = original_argv
                sys.path[:] = original_path

    if set(phases) != {'initial', 'final'} or not effects.is_dir():
        raise ValueError('expected both array boundaries and a published output directory')
    if source_ref != artifact_reference(source, 'scar.python-source') or entry_ref != artifact_reference(entry, 'scar.python-source'):
        raise ValueError('original source changed during observation')
    files = [dict(name=str(path.relative_to(effects)), artifact=artifact_reference(path, 'application/octet-stream'))
             for path in sorted(effects.rglob('*')) if path.is_file()]
    receipt = dict(schema=SCHEMA, schema_version=1, entry=entry_ref, source=source_ref, argv=argv,
                   phases=phases, initial_payload=artifact_reference(initial, 'application/octet-stream'),
                   payload_order=args.fields, **calls, effects=dict(files=files),
                   stdout=artifact_reference(stdout, 'application/octet-stream'),
                   stderr=artifact_reference(stderr, 'application/octet-stream'),
                   cwd=os.getcwd(), environment={key: os.environ[key] for key in _CONTEXT if key in os.environ},
                   interpreter=dict(executable=str(Path(sys.executable).resolve(strict=True)), version=sys.version),
                   launcher=artifact_reference(Path(__file__).resolve(), 'scar.python-source'),
                   instrumented=True, performance_measurement=False)
    with (output / 'observation.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()

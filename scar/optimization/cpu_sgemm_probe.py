"""Compare installed CPU FP32 SGEMM providers on matched native calls."""

import argparse
from array import array
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sysconfig
from tempfile import TemporaryDirectory

from scar.ir.execution_recipe import artifact_reference


SCHEMA = 'scar.cpu-sgemm-provider-observation'
_SOURCE = Path(__file__).resolve().parents[1]/'codegen/cpu_sgemm_probe.cpp.in'
_ENV = dict(LANG='C.UTF-8', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
            OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE', BLIS_NUM_THREADS='1')


@dataclass(frozen=True)
class Provider:
    name: str
    path: Path
    symbol: str = 'cblas_sgemm'
    integer_bits: int = 32
    set_threads: str = '-'
    get_threads: str = '-'


def _installed(libtorch):
    yield Provider('LibTorch ATen', libtorch/'lib/libtorch_cpu.so', symbol='at::mm_out')
    machine = platform.machine()
    for library in sorted(Path('/opt/nvidia/hpc_sdk').glob(f'Linux_{machine}/*/compilers/lib/libblas_lp64.so')):
        version = library.resolve().parents[2].name
        yield Provider(f'NVHPC {version} OpenBLAS LP64', library, set_threads='openblas_set_num_threads',
                       get_threads='openblas_get_num_threads')
    package = Path(sysconfig.get_path('platlib'))
    for wheel, bits, suffix in (('scipy', 32, ''), ('numpy', 64, '64_')):
        for library in sorted((package/(wheel+'.libs')).glob('libscipy_openblas*.so')):
            if ('openblas64_' in library.name) != (bits == 64):
                continue
            yield Provider(f'{wheel} OpenBLAS {bits}-bit integer ABI', library, 'scipy_cblas_sgemm'+suffix, bits,
                           'scipy_openblas_set_num_threads'+suffix, 'scipy_openblas_get_num_threads'+suffix)
    multiarch = sysconfig.get_config_var('MULTIARCH')
    if multiarch:
        yield Provider('System BLAS LP64', Path('/usr/lib')/multiarch/'libblas.so.3')


def _content(path):
    with Path(path).open('rb') as stream:
        digest = sha256()
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest(), size_bytes=Path(path).stat().st_size)


def _compare(reference, output):
    expected = array('f', reference)
    candidate = array('f', output)
    if len(expected) != len(candidate) or any(not isfinite(value) for value in candidate):
        raise ValueError('SGEMM output violates the finite FP32 matrix contract')
    return dict(byte_exact=reference == output,
        unequal_float_values=sum(a != b for a, b in zip(expected, candidate)),
        maximum_absolute_difference=max(abs(a-b) for a, b in zip(expected, candidate)))


def _compile(scratch, libtorch, compiler, env):
    common = [*shlex.split(compiler), '-std=c++20', '-O3', '-x', 'c++', str(_SOURCE), '-x', 'none']
    root = libtorch/'lib'
    builds = {'cblas': ['-ldl'], 'aten': ['-DSCAR_ATEN_PROBE', '-D_GLIBCXX_USE_CXX11_ABI=1',
        '-I'+str(libtorch/'include'), '-Wl,-rpath,'+str(root), '-L'+str(root), '-ltorch_cpu', '-lc10']}
    paths = {}
    for name, arguments in builds.items():
        path = scratch/name
        subprocess.run([*common, *arguments, '-o', str(path)], env=env, check=True, capture_output=True, text=True)
        paths[name] = path
    return paths


def probe(libtorch, left, right, shape, cpus, out, *, warmups=2, repetitions=7, compiler='g++', target=None):
    """Collect warmed operator samples only, with an exact reference comparison."""
    if (len(shape) != 3 or any(type(value) is not int or not 0 < value <= 32768 for value in shape)
            or not cpus or len(cpus) != len(set(cpus)) or not set(cpus) <= os.sched_getaffinity(0)
            or not 1 <= warmups <= 100 or not 3 <= repetitions <= 100):
        raise ValueError('positive matrix dimensions, usable CPUs and bounded probe counts required')
    m, n, k = shape
    if max(m*k, k*n, m*n) > 2**26:
        raise ValueError('matrix payload exceeds the bounded CPU probe budget')
    libtorch, left, right, out = (Path(path).resolve() for path in (libtorch, left, right, out))
    if out.exists() or not (libtorch/'lib/libtorch_cpu.so').is_file():
        raise ValueError('an installed CPU LibTorch root and fresh output are required')
    inputs = [_content(left), _content(right)]
    if [row['size_bytes'] for row in inputs] != [4*m*k, 4*k*n]:
        raise ValueError('input file sizes do not match the dense FP32 row-major contract')
    target_ref = None
    if target is not None:
        from scar.compiler.plan_aten_boundary import _target

        target, _ = _target(target, tuple(cpus), allow_cpu_subset=True)
        target_ref = asdict(target)
    env = dict(_ENV, PATH=os.environ.get('PATH', os.defpath))
    env.update({key: str(len(cpus)) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS')})
    out.mkdir(parents=True)
    rows, seen = [], set()
    launcher = ['/usr/bin/taskset', '-c', ','.join(map(str, cpus))]
    with TemporaryDirectory(prefix='.probe-', dir=out) as temporary:
        scratch = Path(temporary)
        binaries = _compile(scratch, libtorch, compiler, env)
        for provider in _installed(libtorch):
            if not provider.path.is_file():
                continue
            library = provider.path.resolve()
            if library in seen:
                continue
            seen.add(library)
            receipt = _content(library)
            row = dict(library=receipt, provider=provider.name, symbol=provider.symbol,
                       integer_bits=provider.integer_bits if rows else None)
            is_reference = not rows
            output = scratch/'publication.f32'
            path = binaries['aten' if is_reference else 'cblas']
            command = [*launcher, str(path), *map(str, shape), str(len(cpus)), str(warmups), str(repetitions),
                       str(left), str(right), str(output), str(library), provider.symbol, str(provider.integer_bits),
                       provider.set_threads, provider.get_threads]
            run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
            if run.returncode and not is_reference:
                rows.append(dict(row, state='NOT_QUALIFIED', reason=run.stderr.strip()))
                continue
            run.check_returncode()
            observed_threads, *samples = map(int, run.stdout.split())
            if observed_threads != len(cpus) or len(samples) != repetitions or min(samples) <= 0:
                raise ValueError('invalid CPU provider observation')
            raw = output.read_bytes()
            if is_reference:
                reference = raw
            quality = _compare(reference, raw)
            output.unlink()
            if _content(library) != receipt:
                raise ValueError('library bytes changed during SGEMM comparison')
            rows.append(dict(row, state='MEASURED', threads=observed_threads, samples_ns=samples,
                median_ns=statistics.median(samples), lower_sample_ns=min(samples), upper_sample_ns=max(samples),
                output_sha256=sha256(raw).hexdigest(), comparison_to_aten=quality))
    if [_content(left), _content(right)] != inputs:
        raise ValueError('inputs changed during SGEMM comparison')
    measured = [row for row in rows if row['state'] == 'MEASURED']
    def by_median(row):
        return row['median_ns'], row['provider']

    fastest = min(measured, key=by_median)
    exact = [row for row in measured if row['comparison_to_aten']['byte_exact']]
    exact_fastest = min(exact, key=by_median)
    report = dict(schema=SCHEMA, schema_version=1, operator='SGEMM', shape_mnk=list(shape),
        dtype='FP32', layout='CONTIGUOUS_ROW_MAJOR', alpha=1, beta=0,
        cpu_ids=list(cpus), machine=platform.machine(), inputs=inputs, warmups=warmups, repetitions=repetitions,
        sample_scope='WARM_PREALLOCATED_SINGLE_OPERATOR_CALL', target=target_ref,
        source=artifact_reference(_SOURCE, 'scar.cpp-source'), compiler=compiler, providers=rows,
        fastest_observed_provider=fastest['provider'], fastest_byte_exact_observed_provider=exact_fastest['provider'],
        fastest_observed_envelope_separated=all(fastest['upper_sample_ns'] < row['lower_sample_ns']
            for row in measured if row is not fastest),
        byte_exact_scope='THE_PROBED_OPERANDS', full_program_performance_measured=False, deployment_authorized=False)
    (out/'result.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--libtorch', required=True)
    parser.add_argument('--left', required=True)
    parser.add_argument('--right', required=True)
    parser.add_argument('--shape', nargs=3, type=int, required=True, metavar=('M', 'N', 'K'))
    parser.add_argument('--cpus', nargs='+', type=int, required=True)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repetitions', type=int, default=7)
    parser.add_argument('--compiler', default='g++')
    parser.add_argument('--target', help='pin the original hardware graph for use as a scheduling component')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    result = probe(**vars(args))
    print(json.dumps(dict(fastest=result['fastest_observed_provider'],
        fastest_byte_exact=result['fastest_byte_exact_observed_provider'], result=str(Path(args.out)/'result.json')),
        indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

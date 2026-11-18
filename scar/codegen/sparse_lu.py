"""Build an independent SuperLU symbolic reuse study against SciPy's BLAS."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.superlu-symbolic-study'
SOURCE = Path(__file__).with_name('sparse_lu_v1.cpp.in')
PROVIDER_VERSION = '1.18.1'
_INDEX_CONFIGURATION = b'#ifndef SUPERLU_CONFIG_H\n#define SUPERLU_CONFIG_H\ntypedef int int_t;\n#endif\n'
_C_SOURCES = (
    'colamd', 'dcolumn_bmod', 'dcolumn_dfs', 'dcopy_to_ucol', 'dgssv', 'dgstrf', 'dgstrs',
    'dmemory', 'dmyblas2', 'dpanel_bmod', 'dpanel_dfs', 'dpivotL', 'dpruneL',
    'dsnode_bmod', 'dsnode_dfs', 'dsp_blas2', 'dsp_blas3', 'dutil', 'get_perm_c', 'heap_relax_snode',
    'input_error', 'mark_relax', 'memory', 'mmd', 'relax_snode', 'sp_coletree',
    'sp_ienv', 'sp_preorder', 'superlu_timer', 'util',
)


def _installed_carrier():
    import scipy
    from scipy.sparse.linalg._dsolve import _superlu

    if scipy.__version__ != PROVIDER_VERSION:
        raise ValueError('SuperLU study requires the SciPy 1.18.1 provider')
    return artifact_reference(_superlu.__file__, 'scar.native-library-file')


def _wheel(carrier):
    if carrier.get('schema') != 'scar.native-library-file':
        raise ValueError('expected the exact original SuperLU native carrier receipt')
    read_artifact(carrier, json_document=False)
    path = Path(carrier['uri']).resolve()
    if (str(path) != carrier['uri'] or not path.name.startswith('_superlu.') or path.suffix != '.so'
            or path.parent.parts[-4:] != ('scipy', 'sparse', 'linalg', '_dsolve')):
        raise ValueError('SuperLU carrier does not identify the original wheel extension')
    return dict(carrier, schema='scar.provider-binary')


def _needed(path):
    dynamic = subprocess.check_output(['readelf', '-d', str(path)], text=True,
        env={**os.environ, 'LC_ALL': 'C'})
    return set(re.findall(r'\(NEEDED\)\s+Shared library: \[([^\]]+)\]', dynamic))


def _blas(wheel, path):
    original = Path(wheel['uri'])
    if path.parent != original.parents[4] / 'scipy.libs' or path.name not in _needed(original):
        raise ValueError('select the exact BLAS library needed by the original SciPy SuperLU')
    return artifact_reference(path, 'scar.provider-binary')


def validate_symbolic_solve(native, carrier):
    """Check a sealed or newly built bundle against the original carrier."""
    bundle = read_artifact(native)
    wheel = _wheel(carrier)
    directory = Path(native['uri']).parent
    if ((bundle['schema_version'], bundle['abi_version'], bundle['language'], bundle['scope'])
            != (1, 1, 'c++20', 'EXPLICIT_STUDY') or bundle['provider_version'] != PROVIDER_VERSION
            or bundle['wheel'] != wheel or bundle['source']['uri'] != str(directory / 'sparse_lu.cpp')
            or bundle['library']['uri'] != str(directory / 'sparse_lu.so')
            or bundle['configuration']['uri'] != str(directory / 'superlu_config.h')
            or bundle['template'] != artifact_reference(SOURCE, 'scar.cpp-source')
            or read_artifact(bundle['source'], json_document=False) != SOURCE.read_bytes()
            or read_artifact(bundle['configuration'], json_document=False) != _INDEX_CONFIGURATION):
        raise ValueError('SuperLU study artifact or original carrier differs')
    blas = Path(bundle['blas']['uri']).resolve()
    if bundle['blas'] != _blas(wheel, blas):
        raise ValueError('SuperLU study BLAS receipt differs from the original carrier dependency')
    for reference in (*bundle['vendor'], bundle['library']):
        read_artifact(reference, json_document=False)
    if blas.name not in _needed(bundle['library']['uri']):
        raise ValueError('SuperLU study library has a different linked BLAS')
    return bundle


def build_symbolic_solve(source_root, out, *, blas, jobs=4, carrier=None):
    """Compile retained v1.18.1 source using the admitted original extension."""
    root, out, blas = (Path(path).resolve() for path in (source_root, out, blas))
    if out.exists() or type(jobs) is not int or jobs <= 0:
        raise ValueError('fresh SuperLU build directory and positive job count required')
    wheel = _wheel(_installed_carrier() if carrier is None else carrier)
    blas_ref = _blas(wheel, blas)
    upstream = root / 'SuperLU' / 'SRC'
    source_files = [upstream / f'{name}.c' for name in _C_SOURCES]
    headers = sorted(upstream.glob('*.h'))
    configs = [root / name for name in ('scipy_slu_config.h', 'scipy_slu_blas_config.h')]
    vendor = [artifact_reference(path, 'scar.provider-source')
        for path in (*source_files, *headers, *configs, upstream / 'superlu_config.h.in',
            root / 'meson.build', root / 'SuperLU' / 'License.txt')]
    cc, cxx = (shlex.split(os.environ.get(name, default))
        for name, default in (('CC', 'cc'), ('CXX', 'c++')))
    if not cc or not cxx:
        raise ValueError('C and C++20 compilers required')

    out.mkdir(parents=True)
    config = out / 'superlu_config.h'
    config.write_bytes(_INDEX_CONFIGURATION)
    generated = out / 'sparse_lu.cpp'
    generated.write_bytes(SOURCE.read_bytes())
    library = out / 'sparse_lu.so'
    include = ['-I', str(out), '-I', str(upstream), '-I', str(root)]
    shared = ['-O3', '-fPIC', '-fvisibility=hidden', '-fno-fast-math', *include]
    c_flags = ['-std=c11', '-D_GNU_SOURCE', '-DUSE_VENDOR_BLAS=1', '-DBLAS_SYMBOL_PREFIX=scipy_', *shared]

    def compile_c(source):
        object_path = out / f'{source.stem}.o'
        subprocess.run([*cc, *c_flags, '-c', str(source), '-o', str(object_path)], check=True)
        return str(object_path)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        objects = list(pool.map(compile_c, source_files))
    cpp_flags = ['-std=c++20', *shared]
    link = ['-shared', '-Wl,-z,defs', f'-Wl,-rpath,{blas.parent}', str(blas), '-lm']
    subprocess.run([*cxx, *cpp_flags, str(generated), *objects, *link, '-o', str(library)], check=True)
    if blas.name not in _needed(library):
        raise ValueError('SuperLU study library did not link the original BLAS dependency')
    for reference in (*vendor, wheel, blas_ref):
        read_artifact(reference, json_document=False)
    manifest = dict(schema=SCHEMA, schema_version=1, abi_version=1, language='c++20',
        scope='EXPLICIT_STUDY', provider_version=PROVIDER_VERSION,
        wheel=wheel, blas=blas_ref, vendor=vendor,
        source=artifact_reference(generated, 'scar.cpp-source'),
        configuration=artifact_reference(config, 'scar.provider-source'),
        template=artifact_reference(SOURCE, 'scar.cpp-source'),
        library=artifact_reference(library, 'scar.native-library'),
        compiler=dict(cc=cc, cxx=cxx, c_flags=c_flags, cpp_flags=cpp_flags))
    with (out / 'artifact.json').open('x') as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return manifest

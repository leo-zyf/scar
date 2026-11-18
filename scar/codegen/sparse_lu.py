"""Build an independent SuperLU symbolic reuse study against SciPy's BLAS."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess

from scar.ir.execution_recipe import artifact_reference


SCHEMA = 'scar.superlu-symbolic-study'
SOURCE = Path(__file__).with_name('sparse_lu_v1.cpp.in')
_C_SOURCES = (
    'colamd', 'dcolumn_bmod', 'dcolumn_dfs', 'dcopy_to_ucol', 'dgssv', 'dgstrf', 'dgstrs',
    'dmemory', 'dmyblas2', 'dpanel_bmod', 'dpanel_dfs', 'dpivotL', 'dpruneL',
    'dsnode_bmod', 'dsnode_dfs', 'dsp_blas2', 'dsp_blas3', 'dutil', 'get_perm_c', 'heap_relax_snode',
    'input_error', 'mark_relax', 'memory', 'mmd', 'relax_snode', 'sp_coletree',
    'sp_ienv', 'sp_preorder', 'superlu_timer', 'util',
)


def build_symbolic_solve(source_root, out, *, blas, jobs=4):
    """Compile the caller's retained SciPy v1.18.1 SuperLU source for study."""
    import scipy
    from scipy.sparse.linalg._dsolve import _superlu

    if scipy.__version__ != '1.18.1':
        raise ValueError('SuperLU study requires the SciPy 1.18.1 provider')
    root, out, blas = (Path(path).resolve() for path in (source_root, out, blas))
    if out.exists() or type(jobs) is not int or jobs <= 0:
        raise ValueError('fresh SuperLU build directory and positive job count required')
    depends = subprocess.check_output(['readelf', '-d', _superlu.__file__], text=True)
    if blas.parent != Path(scipy.__file__).resolve().parents[1] / 'scipy.libs' or f'[{blas.name}]' not in depends:
        raise ValueError('select the exact BLAS library needed by the installed SciPy SuperLU')
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
    config.write_text('#ifndef SUPERLU_CONFIG_H\n#define SUPERLU_CONFIG_H\ntypedef int int_t;\n#endif\n')
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
    manifest = dict(schema=SCHEMA, schema_version=1, abi_version=1, language='c++20',
        scope='EXPLICIT_STUDY', provider_version=scipy.__version__,
        wheel=artifact_reference(_superlu.__file__, 'scar.provider-binary'),
        blas=artifact_reference(blas, 'scar.provider-binary'), vendor=vendor,
        source=artifact_reference(generated, 'scar.cpp-source'),
        configuration=artifact_reference(config, 'scar.provider-source'),
        template=artifact_reference(SOURCE, 'scar.cpp-source'),
        library=artifact_reference(library, 'scar.native-library'),
        compiler=dict(cc=cc, cxx=cxx, c_flags=c_flags, cpp_flags=cpp_flags))
    with (out / 'artifact.json').open('x') as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return manifest

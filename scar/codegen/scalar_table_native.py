"""Build and verify the C++20 exact built-in scalar table study.

The native primitive reads current inputs and returns NotImplemented before
invoking unsupported behavior. The caller owns the source and effect proof
and must preserve the original expression for this decline.
"""

from importlib.machinery import ExtensionFileLoader
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import sysconfig
from tempfile import TemporaryDirectory

from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.scalar-table-native-study'
ABI_VERSION = 1
MODULE_NAME = '_scar_scalar_table_v1'
TEMPLATE = Path(__file__).resolve().with_name('scalar_table_v1.cpp.in')
MAX_LIVE_KEYS = 32
COMPILER_OPTIONS = ('-std=c++20', '-O3', '-shared', '-fPIC', '-fvisibility=hidden')
_SOURCE = 'scalar_table.cpp'
_SCOPE = 'EXPLICIT_STUDY_EXTERNAL_ADMISSION'


def _python_target():
    extension = sysconfig.get_config_var('EXT_SUFFIX')
    if sys.implementation.name != 'cpython' or not extension or sysconfig.get_config_var('Py_GIL_DISABLED'):
        raise ValueError('scalar table lowering requires the ordinary CPython extension ABI and GIL')
    return dict(version=sys.version, machine=platform.machine(), gil='REQUIRED',
                soabi=sysconfig.get_config_var('SOABI'), extension_suffix=extension,
                executable=artifact_reference(Path(sys.executable).resolve(), 'scar.native-executable-file'))


def build_scalar_table(out):
    """Build a fresh native study and return its manifest artifact reference."""
    directory = Path(out).resolve()
    if directory.exists():
        raise FileExistsError(directory)
    target = _python_target()
    extension = MODULE_NAME + target['extension_suffix']
    include = Path(sysconfig.get_path('include')).resolve()
    headers = [artifact_reference(include / name, 'scar.cpp-source') for name in ('Python.h', 'pyconfig.h')]
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    if not compiler:
        raise ValueError('CXX must select an available C++20 compiler')
    version = subprocess.check_output([*compiler, '--version'], text=True).splitlines()[0]
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scalar-table-', dir=directory.parent) as pending:
        build = Path(pending)
        source = build / _SOURCE
        source.write_bytes(TEMPLATE.read_bytes())
        subprocess.run([*compiler, *COMPILER_OPTIONS, '-I', str(include), str(source),
                        '-o', str(build / extension)], check=True)
        manifest = dict(schema=SCHEMA, schema_version=1, abi_version=ABI_VERSION,
                        language='c++20', scope=_SCOPE, python=target, headers=headers,
                        template=artifact_reference(TEMPLATE, 'scar.cpp-source'),
                        source=dict(artifact_reference(source, 'scar.cpp-source'), uri=str(directory / _SOURCE)),
                        library=dict(artifact_reference(build / extension, 'scar.native-library-file'),
                                     uri=str(directory / extension)),
                        compiler=dict(command=compiler, version=version, options=list(COMPILER_OPTIONS)),
                        module=MODULE_NAME, entrypoint='subtract', max_live_keys=MAX_LIVE_KEYS,
                        dictionary_key_proof='LIVE_EXACT_STR_SCAN', retention='NONE', deployment_authorized=False)
        (build / 'artifact.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
        if directory.exists():
            raise FileExistsError(directory)
        build.rename(directory)
    return artifact_reference(directory / 'artifact.json', SCHEMA)


def load_scalar_table(reference):
    """Verify the sealed CPython, source and library identity before loading."""
    manifest = read_artifact(reference)
    target = _python_target()
    directory = Path(reference['uri']).parent
    library = directory / (MODULE_NAME + target['extension_suffix'])
    if (manifest.get('schema') != SCHEMA or manifest.get('schema_version') != 1
            or manifest.get('language') != 'c++20' or manifest.get('abi_version') != ABI_VERSION
            or manifest.get('scope') != _SCOPE or manifest.get('python') != target
            or manifest.get('module') != MODULE_NAME or manifest.get('entrypoint') != 'subtract'
            or manifest.get('max_live_keys') != MAX_LIVE_KEYS or manifest.get('retention') != 'NONE'
            or manifest.get('dictionary_key_proof') != 'LIVE_EXACT_STR_SCAN'
            or manifest.get('deployment_authorized') is not False
            or manifest.get('template') != artifact_reference(TEMPLATE, 'scar.cpp-source')
            or manifest['source']['uri'] != str(directory / _SOURCE)
            or manifest['library']['uri'] != str(library)
            or read_artifact(manifest['source'], json_document=False) != TEMPLATE.read_bytes()):
        raise ValueError('the scalar table study or CPython target differs')
    read_artifact(manifest['library'], json_document=False)
    loader = ExtensionFileLoader(MODULE_NAME, str(library))
    spec = spec_from_file_location(MODULE_NAME, library, loader=loader)
    module = module_from_spec(spec)
    loader.exec_module(module)
    if (module.scar_operator_abi != ABI_VERSION or module.max_live_keys != MAX_LIVE_KEYS
            or not callable(module.subtract)):
        raise ValueError('the scalar table study operator ABI differs')
    return module


__all__ = ['build_scalar_table', 'load_scalar_table']

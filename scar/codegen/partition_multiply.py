"""Build a CPython partition-orchestration study without a new numeric kernel.

Callers must admit the source region, NumPy provider and actual arguments.
The copy-eliding entry additionally requires a writable disjoint output,
equivalent dtype/ufunc dispatch on ndarray views, complete disjoint slice
coverage, and a contract that permits the changed intermediate output state.
Both entries propagate provider errors, including errors after writes.
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


SCHEMA = 'scar.partition-multiply-native-study'
ABI_VERSION = 1
MODULE_NAME = '_scar_partition_multiply_v1'
TEMPLATE = Path(__file__).resolve().with_name('partition_multiply_v1.cpp.in')
COMPILER_OPTIONS = ('-std=c++20', '-O3', '-shared', '-fPIC', '-fvisibility=hidden')
ENTRYPOINTS = ('apply', 'apply_without_copy')
_SCOPE = 'EXPLICIT_STUDY_EXTERNAL_ADMISSION'
_SOURCE = 'partition_multiply.cpp'


def _image_stamp(path):
    if sys.platform != 'linux':
        return None
    try:
        active = os.stat('/proc/self/exe')
    except OSError:
        return None
    on_disk = path.stat()
    stamp = lambda entry: [entry.st_dev, entry.st_ino, entry.st_size, entry.st_mtime_ns, entry.st_ctime_ns]
    current = stamp(active)
    if current != stamp(on_disk):
        raise ValueError('running Python image differs from its executable path')
    return current


def verify_running_python(reference):
    """Hash the exact running image once for directly spawned Linux workers."""
    executable = Path(sys.executable).resolve()
    before = _image_stamp(executable)
    if before is None:
        return None
    expected = read_artifact(reference)['python']['executable']
    actual = artifact_reference(executable, 'scar.native-executable-file')
    if actual != expected or _image_stamp(executable) != before:
        raise ValueError('parent Python executable differs from the compiled target')
    return dict(executable=actual, live_image_stamp=before)


def _python_target(parent_image=None):
    if sys.implementation.name != 'cpython' or not sysconfig.get_config_var('EXT_SUFFIX'):
        raise ValueError('the partition study requires the exact CPython extension ABI')
    path = Path(sys.executable).resolve()
    stamp = _image_stamp(path)
    executable = None
    if parent_image is not None:
        if (stamp is not None and stamp == parent_image['live_image_stamp']
                and parent_image['executable']['uri'] == str(path)):
            executable = parent_image['executable']
    if executable is None:
        executable = artifact_reference(path, 'scar.native-executable-file')
        if _image_stamp(path) != stamp:
            raise ValueError('running Python image changed during identity check')
    return dict(version=sys.version, machine=platform.machine(),
                soabi=sysconfig.get_config_var('SOABI'),
                extension_suffix=sysconfig.get_config_var('EXT_SUFFIX'),
                executable=executable)


def build_partition_multiply(out):
    """Publish a fresh, scoped extension build and return its manifest reference."""
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    python = _python_target()
    extension = MODULE_NAME + python['extension_suffix']
    include = Path(sysconfig.get_path('include')).resolve()
    headers = [artifact_reference(include / name, 'scar.cpp-source') for name in ('Python.h', 'pyconfig.h')]
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    if not compiler:
        raise ValueError('CXX must select an available C++20 compiler')
    version = subprocess.check_output([*compiler, '--version'], text=True).splitlines()[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.partition-multiply-', dir=out.parent) as pending:
        directory = Path(pending)
        source = directory / _SOURCE
        source.write_bytes(TEMPLATE.read_bytes())
        subprocess.run([*compiler, *COMPILER_OPTIONS, '-I', str(include), str(source),
                        '-o', str(directory / extension)], check=True)
        manifest = dict(schema=SCHEMA, schema_version=1, abi_version=ABI_VERSION,
                        language='c++20', scope=_SCOPE, python=python, headers=headers,
                        template=artifact_reference(TEMPLATE, 'scar.cpp-source'),
                        source=dict(artifact_reference(source, 'scar.cpp-source'), uri=str(out / _SOURCE)),
                        library=dict(artifact_reference(directory / extension, 'scar.native-library-file'),
                                     uri=str(out / extension)),
                        compiler=dict(command=compiler, version=version, options=list(COMPILER_OPTIONS)),
                        module=MODULE_NAME, entrypoints=list(ENTRYPOINTS),
                        numerical_provider='EXTERNAL_LIVE_PYOBJECT_DISPATCH',
                        deployment_authorized=False)
        (directory / 'artifact.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
        if out.exists():
            raise FileExistsError(out)
        directory.rename(out)
    return artifact_reference(out / 'artifact.json', SCHEMA)


def load_partition_multiply(reference, *, parent_image=None):
    """Verify byte pins and load the internal study without installing a hook."""
    manifest = read_artifact(reference)
    target = _python_target(parent_image)
    directory = Path(reference['uri']).parent
    library = directory / (MODULE_NAME + target['extension_suffix'])
    if (manifest.get('schema') != SCHEMA or manifest.get('schema_version') != 1
            or manifest.get('abi_version') != ABI_VERSION or manifest.get('language') != 'c++20'
            or manifest.get('scope') != _SCOPE or manifest.get('python') != target
            or manifest.get('module') != MODULE_NAME or manifest.get('entrypoints') != list(ENTRYPOINTS)
            or manifest.get('numerical_provider') != 'EXTERNAL_LIVE_PYOBJECT_DISPATCH'
            or manifest.get('deployment_authorized') is not False
            or manifest.get('template') != artifact_reference(TEMPLATE, 'scar.cpp-source')
            or manifest['source']['uri'] != str(directory / _SOURCE)
            or manifest['library']['uri'] != str(library)
            or read_artifact(manifest['source'], json_document=False) != TEMPLATE.read_bytes()):
        raise ValueError('the partition study or CPython target differs')
    read_artifact(manifest['library'], json_document=False)
    loader = ExtensionFileLoader(MODULE_NAME, str(library))
    spec = spec_from_file_location(MODULE_NAME, library, loader=loader)
    module = module_from_spec(spec)
    loader.exec_module(module)
    if module.scar_operator_abi != ABI_VERSION or any(not callable(getattr(module, name)) for name in ENTRYPOINTS):
        raise ValueError('the partition study operator ABI differs')
    return module


__all__ = ['build_partition_multiply', 'load_partition_multiply']

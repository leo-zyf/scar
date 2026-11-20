"""Opt-in NumPy owner allocation epochs for a bounded original capture."""
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig

from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.numpy-allocation-epoch-tracker'
SOURCE = Path(__file__).resolve().parents[1] / 'native/numpy_allocation_epochs.cpp'
MODULE = '_scar_numpy_allocation_epochs'
MAX_LIVE = 16384


def _runtime():
    import numpy as np

    if (sys.platform != 'linux' or sys.implementation.name != 'cpython'
            or sysconfig.get_config_var('Py_GIL_DISABLED') or np.__version__.split('.')[0] != '2'):
        raise ValueError('allocation epochs require Linux GIL-enabled CPython and NumPy 2.x')
    return dict(python_cache_tag=sys.implementation.cache_tag, numpy_version=np.__version__,
                extension_suffix=sysconfig.get_config_var('EXT_SUFFIX'))


def build_tracker(out):
    """Build once outside the child. The recorder only loads the pinned helper."""
    import numpy as np
    runtime = _runtime()
    out = Path(out).resolve()
    binary = out / (MODULE + runtime['extension_suffix'])
    compiler = shlex.split(os.environ.get('CXX', 'g++'))
    command = (*compiler, '-std=c++20', '-O2', '-shared', '-fPIC', '-pthread',
               '-I' + sysconfig.get_path('include'), '-I' + np.get_include(), str(SOURCE), '-o', str(binary))
    out.mkdir(exist_ok=False)
    subprocess.run(command, check=True, timeout=60)
    report = dict(schema=SCHEMA, schema_version=1, runtime=runtime, max_live_allocations=MAX_LIVE,
                  producer=artifact_reference(SOURCE, 'scar.cpp-source'),
                  binary=artifact_reference(binary, 'scar.cpython-extension'))
    manifest = out / 'tracker.json'
    with manifest.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write('\n')
    return manifest


def read_tracker(path):
    reference = artifact_reference(Path(path).resolve(), SCHEMA)
    report = read_artifact(reference)
    binary = Path(report['binary']['uri'])
    if (report['schema_version'] != 1 or report['runtime'] != _runtime()
            or report['max_live_allocations'] != MAX_LIVE
            or report['producer'] != artifact_reference(SOURCE, 'scar.cpp-source')
            or binary.parent != Path(reference['uri']).parent
            or binary.name != MODULE + report['runtime']['extension_suffix']):
        raise ValueError('allocation tracker source, runtime or output identity differs')
    read_artifact(report['binary'], json_document=False)
    return reference, report


class AllocationEpochTracker:
    def __init__(self, manifest):
        self.reference, report = read_tracker(manifest)
        spec = spec_from_file_location(MODULE, report['binary']['uri'])
        self.module = module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.capsule = self.module.start(report['max_live_allocations'])

    def __call__(self, owner):
        if self.capsule is None:
            return None
        result = self.module.lookup(self.capsule, owner)
        if result is None:
            return None
        epoch, cursor = result
        return dict(epoch=epoch, cursor=cursor)

    def close(self):
        if self.capsule is not None:
            self.module.stop(self.capsule)
            self.capsule = None

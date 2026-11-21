"""Load a sealed live partition module without importing the compiler."""

from importlib.machinery import ExtensionFileLoader
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sysconfig

from scar.ir.execution_recipe import read_artifact


EXTENSION_SCHEMA = 'scar.native-numpy-partition-live-extension'
MODULE = '_scar_numpy_partition_live_v1'
ABI_VERSION = 1


def load(reference):
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    filename = reference['uri']
    if reference['schema'] != EXTENSION_SCHEMA or Path(filename).name != MODULE + suffix:
        raise ValueError('live partition extension module or CPython ABI differs')
    read_artifact(reference, json_document=False)
    loader = ExtensionFileLoader(MODULE, filename)
    spec = spec_from_file_location(MODULE, filename, loader=loader)
    module = module_from_spec(spec)
    loader.exec_module(module)
    if module.scar_operator_abi != ABI_VERSION or not callable(module.try_apply):
        raise ValueError('compiled live partition operator ABI differs')
    return module

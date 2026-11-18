"""Compile the original-storage CSR row and diagonal kernel."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import CodeType

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.trace.code_snapshot import code_key

SCHEMA = 'scar.native-csr-enforce'
SCHEMA_VERSION = 2
TOPOLOGY_FEATURE = 'scoped-topology-v1'
SOURCE = Path(__file__).with_name('sparse_csr_v1.cpp.in')
OPTIONS = ('-std=c++20', '-O3', '-shared', '-fPIC', '-fvisibility=hidden', '-fno-fast-math')


def _source_code_keys(modules, bindings):
    keys = {}
    for name, reference in modules:
        source = read_artifact(reference, json_document=False)
        pending = [compile(source, reference['uri'], 'exec', dont_inherit=True, optimize=sys.flags.optimize)]
        while pending:
            code = pending.pop()
            binding = f'{name}:{code.co_qualname}'
            if binding in bindings:
                keys[binding] = code_key(code)
            pending.extend(item for item in code.co_consts if type(item) is CodeType)
    return keys


def build_csr_enforce(out):
    """Build once into a fresh directory. Loading and guarded use are separate."""
    from scar.runtime.sparse_csr import _provider

    directory = Path(out).resolve()
    if directory.exists():
        raise ValueError('fresh CSR C++ output required')
    provider, *_ = _provider(compile_sources=_source_code_keys)
    compiler = shlex.split(os.environ.get('CXX', 'g++'))
    if not compiler:
        raise ValueError('C++20 compiler required')
    directory.mkdir(parents=True)
    source = directory/'sparse_csr.cpp'
    library = directory/'sparse_csr.so'
    source.write_bytes(SOURCE.read_bytes())
    command = [*compiler, *OPTIONS, str(source), '-o', str(library)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    manifest = dict(schema=SCHEMA, schema_version=SCHEMA_VERSION, abi_version=1, language='c++20',
        features=[TOPOLOGY_FEATURE],
        source=artifact_reference(source, 'scar.cpp-source'),
        library=artifact_reference(library, 'scar.native-library'),
        template=artifact_reference(SOURCE, 'scar.cpp-source'),
        compiler=dict(command=compiler, options=list(OPTIONS)), provider=provider)
    with (directory/'artifact.json').open('x') as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return manifest

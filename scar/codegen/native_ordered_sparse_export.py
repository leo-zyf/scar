"""Lower the ordered sparse recipe to a standalone C++20 dispatch package."""

import argparse
import json
import os
from pathlib import Path
import shlex
from shutil import copytree
import subprocess
from tempfile import TemporaryDirectory

from scar.analysis.replay_transaction import replay_transaction
from scar.codegen.sparse_native_export import check_sparse_export
from scar.ir.execution_recipe import SCHEMA as RECIPE, artifact_reference, read_artifact
from scar.optimization.native_ordered_resources import SCHEMA as PLAN, project_native_ordered_resources
from scar.optimization.native_ordered_stream import (
    SCHEMA as STREAM, project_native_ordered_stream, validate_native_ordered_stream,
)
from scar.optimization.sparse_parallel_recipe import verify_sparse_parallel_recipe


SCHEMA = 'scar.native-ordered-sparse-export'
_ABI_HEADER = 'scar_sparse_provider_v1.hpp'
_ORDERED_HEADER = 'scar_ordered_sparse_v1.hpp'
_DISPATCH_HEADER = 'native_ordered_dispatch_v1.h'
_DISPATCH_SOURCE = Path(__file__).with_name(_DISPATCH_HEADER + '.in')
_ORDERED_SOURCE = Path(__file__).with_name('native_ordered_sparse_v1.h.in')
_MAIN_SOURCE = Path(__file__).with_name('native_ordered_sparse_main_v1.cpp.in')
_STREAM_PLAN = 'stream.json'
_STREAM_MAIN = 'scar_ordered_sparse.cpp'
_STREAM_EXECUTABLE = 'scar_ordered_sparse'
_CXXFLAGS = ('-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic', '-Wl,-z,defs')


def _render(plan, blas):
    case, placement = plan['case'], plan['placement']
    lanes = [placement['owner'], *placement['workers']]
    identities = [(row['cpu_id'], row['package_id'], -1 if row['die_id'] is None else row['die_id'], row['core_id'])
                  for row in lanes]
    memory = plan['memory']
    bindings = dict(MATRIX_SIZE=case['size'], MATRIX_NNZ=case['nnz'], REQUEST_COUNT=case['call_count'],
        NUMA_NODE=placement['cpu_numa_node'], OWNER_CPU=placement['owner_cpu_id'],
        WORKER_CPUS=', '.join(map(str, placement['worker_cpu_ids'])),
        CPU_PLACEMENT=', '.join('{' + ', '.join(map(str, identity)) + '}' for identity in identities),
        WINDOW=plan['dispatch']['window'], WORKER_LIMIT=memory['worker_address_space_limit_bytes'],
        SYMBOLIC_BUDGET=memory['per_worker_symbolic_budget_bytes'],
        MEMORY_HEADROOM=memory['minimum_launch_headroom_bytes'],
        PAYLOAD_BUDGET=memory['owner_pending_input_bytes'] + memory['owner_pending_result_bound_bytes'],
        BLAS_MEMBER=json.dumps(blas))
    source = _ORDERED_SOURCE.read_text()
    for field, value in bindings.items():
        source = source.replace('@SCAR_' + field + '@', str(value))
    return source.encode()


def _local(root, name, schema):
    path = root / name
    if path.resolve(strict=True) != path:
        raise ValueError('expected a regular package member: ' + name)
    return {**artifact_reference(path, schema), 'uri': name}


def _render_main(stream):
    digest = stream['recipe_id'].removeprefix('recipe-v1:')
    return _MAIN_SOURCE.read_text().replace('@SCAR_RECIPE_DIGEST@', digest).encode()


def _write_stream(directory, plan):
    stream = project_native_ordered_stream(plan)
    (directory / _STREAM_PLAN).write_text(json.dumps(stream, sort_keys=True, indent=2, allow_nan=False) + '\n')
    (directory / _STREAM_MAIN).write_bytes(_render_main(stream))
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    if not compiler:
        raise ValueError('CXX must select an available C++20 compiler')
    version = subprocess.check_output([*compiler, '--version'], text=True).splitlines()[0]
    invocation = [*compiler, *_CXXFLAGS, _STREAM_MAIN, '-ldl', '-o', _STREAM_EXECUTABLE]
    subprocess.run(invocation, cwd=directory, check=True)
    return dict(protocol=_local(directory, _STREAM_PLAN, STREAM),
        source=_local(directory, _STREAM_MAIN, 'scar.cpp-source'),
        executable=_local(directory, _STREAM_EXECUTABLE, 'scar.native-executable-file'),
        compiler=dict(command=compiler, version=version, options=list(_CXXFLAGS)))


def _check_stream(directory, plan, transport):
    if (transport['protocol'] != _local(directory, _STREAM_PLAN, STREAM)
            or transport['source'] != _local(directory, _STREAM_MAIN, 'scar.cpp-source')
            or transport['executable'] != _local(directory, _STREAM_EXECUTABLE, 'scar.native-executable-file')
            or transport['compiler']['options'] != list(_CXXFLAGS)
            or not os.access(directory / _STREAM_EXECUTABLE, os.X_OK)):
        raise ValueError('native stream executable or protocol differs from its receipt')
    protocol = read_artifact({**transport['protocol'], 'uri': str(directory / _STREAM_PLAN)})
    validate_native_ordered_stream(plan, protocol)
    if (directory / _STREAM_MAIN).read_bytes() != _render_main(protocol):
        raise ValueError('native stream entry differs from its recipe and protocol')


@replay_transaction()
def export_native_ordered(recipe, provider, out):
    recipe_ref = artifact_reference(recipe, RECIPE)
    source = verify_sparse_parallel_recipe(read_artifact(recipe_ref))
    plan = project_native_ordered_resources(source)
    provider = Path(provider).resolve(strict=True)
    native = check_sparse_export(provider)
    capability = source['implementation']['backend_capability_record']
    provenance = native['build_provenance']
    if (provenance['recipe'] != capability['serial_recipe']
            or provenance['lowering'] != capability['serial_lowering']
            or provenance['native'] != capability['native'] or native['abi']['version'] != capability['abi_version']):
        raise ValueError('ordered recipe and standalone numeric provider do not share the same lowered ABI')
    closure_path = provider / native['provider']['uri']
    closure = read_artifact({**native['provider'], 'uri': str(closure_path)})
    matches = [row for row in closure['members'] if row['origin'] == capability['blas']['uri']
               and row['sha256'] == capability['blas']['sha256']]
    if len(matches) != 1:
        raise ValueError('ordered recipe requires its exact single-thread BLAS member')
    blas = 'provider/' + matches[0]['path']
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix='.scar-native-ordered-', dir=out.parent) as temporary:
        directory = Path(temporary)
        copytree(provider, directory / 'numeric')
        (directory / _ABI_HEADER).write_bytes((provider / native['header']['uri']).read_bytes())
        (directory / _DISPATCH_HEADER).write_bytes(_DISPATCH_SOURCE.read_bytes())
        (directory / _ORDERED_HEADER).write_bytes(_render(plan, blas))
        (directory / 'plan.json').write_text(json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + '\n')
        stream = _write_stream(directory, plan)
        manifest = dict(schema=SCHEMA, schema_version=2, recipe=recipe_ref, recipe_id=source['recipe_id'],
            numeric=_local(directory, 'numeric/artifact.json', native['schema']), blas_member=blas,
            plan=_local(directory, 'plan.json', PLAN), headers=[_local(directory, name, 'scar.cpp-header')
                for name in (_ABI_HEADER, _DISPATCH_HEADER, _ORDERED_HEADER)],
            caller_boundary='native CSR structure and one binary64 value/RHS frame per ordered request',
            failure='first failed request stops publication; original Python entry is separate',
            stream=stream, cost='unmeasured native stream boundary', deployment_authorized=False)
        (directory / 'artifact.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
        check_native_ordered_export(directory)
        if out.exists():
            raise FileExistsError(out)
        directory.rename(out)
    return artifact_reference(out / 'artifact.json', SCHEMA)


def check_native_ordered_export(path):
    root = Path(path).resolve(strict=True)
    manifest = read_artifact(artifact_reference(root / 'artifact.json', SCHEMA))
    verification = check_sparse_export(root / 'numeric')
    expected_headers = [_local(root, name, 'scar.cpp-header') for name in (_ABI_HEADER, _DISPATCH_HEADER, _ORDERED_HEADER)]
    version = manifest.get('schema_version')
    if (version not in (1, 2) or ('stream' in manifest) != (version == 2) or manifest['deployment_authorized'] is not False
            or manifest['headers'] != expected_headers
            or manifest['numeric'] != _local(root, 'numeric/artifact.json', verification['schema'])
            or manifest['plan'] != _local(root, 'plan.json', PLAN)):
        raise ValueError('an ordered package member differs from its receipt')
    plan = read_artifact({**manifest['plan'], 'uri': str(root / 'plan.json')})
    closure_path = root / 'numeric' / verification['provider']['uri']
    closure = read_artifact({**verification['provider'], 'uri': str(closure_path)})
    operator, = [row for row in closure['members'] if row['path'] == closure['root']]
    direct = set(operator['elf']['needed']) - set(closure['system_sonames'])
    blas = [row['path'] for row in closure['members'] if row['elf']['soname'] in direct]
    if (len(blas) != 1 or manifest['blas_member'] != 'provider/' + blas[0]
            or manifest['recipe_id'] != plan['recipe_id']
            or (root / _ABI_HEADER).read_bytes() != (root / 'numeric' / verification['header']['uri']).read_bytes()
            or (root / _DISPATCH_HEADER).read_bytes() != _DISPATCH_SOURCE.read_bytes()
            or (root / _ORDERED_HEADER).read_bytes() != _render(plan, manifest['blas_member'])):
        raise ValueError('standalone C++ scheduling does not match its native plan')
    if version == 2:
        _check_stream(root, plan, manifest['stream'])
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe')
    parser.add_argument('--provider')
    parser.add_argument('--out')
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    export = (args.recipe, args.provider, args.out)
    if args.check and not any(export):
        result = check_native_ordered_export(args.check)
        print(json.dumps(dict(verified=True, recipe_id=result['recipe_id'])))
    elif not args.check and all(export):
        print(json.dumps(export_native_ordered(*export)))
    else:
        parser.error('choose --check PACKAGE or supply --recipe, --provider and --out')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

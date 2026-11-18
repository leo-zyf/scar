"""Export a verified sparse lowering as a relocatable C++ ABI consumer package."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scar.analysis.replay_transaction import replay_transaction
from scar.codegen.native_provider_bundle import SCHEMA as PROVIDER, build_native_provider_bundle, validate_native_provider_bundle
from scar.codegen.sparse_recipe import SCHEMA as LOWERING, verify_sparse_lowering
from scar.ir.execution_recipe import SCHEMA as RECIPE, artifact_reference, read_artifact


SCHEMA = 'scar.sparse-native-export'
_SYSTEM_LIBRARIES = ('ld-linux-x86-64.so.2', 'libc.so.6', 'libgcc_s.so.1', 'libm.so.6',
                     'libpthread.so.0', 'libstdc++.so.6', 'libz.so.1')
_HEADER = 'scar_sparse_provider_v1.hpp'
_PROVIDER_MANIFEST = 'provider/bundle.json'
_TEMPLATE = Path(__file__).with_name('sparse_native_consumer_v1.h.in')
_ABI = dict(language='c++20', version=1, calling_convention='C', loader='linux-glibc',
    matrix='square CSR with sorted, unique int32 column indices and int32 row offsets',
    numeric='IEEE binary64 values, dense RHS and output, round-to-nearest',
    structure='owned by symbolic handle and verified against each solve',
    publish='caller-owned private output; publish only on zero status',
    provider_failure_diagnostics='unqualified',
    interpreter_required=False, blas_threads='external setting',
    symbolic_memory='retained symbolic structure only; provider scratch and NUMA unresolved')


def _render(bundle):
    by_path = {row['path']: row['elf']['soname'] for row in bundle['members']}
    order = [*bundle['preload'], bundle['root']]
    paths = [(f'provider/{relative}', by_path[relative] or '') for relative in order]
    rows = ['            {' + ', '.join(map(json.dumps, row)) + '},' for row in paths]
    return _TEMPLATE.read_text().replace('@SCAR_PROVIDER_LOAD_ORDER@', '\n'.join(rows)).encode()


def _local_artifact(root, relative, schema):
    path = root / relative
    if path.resolve(strict=True) != path:
        raise ValueError('a local regular package member is required: ' + relative)
    actual = artifact_reference(path, schema)
    return {**actual, 'uri': relative}


@replay_transaction()
def export_sparse_native(recipe, lowering, out):
    """Use the original graph and provider only as input to a fresh native export."""
    recipe_ref = artifact_reference(recipe, RECIPE)
    lowering_ref = artifact_reference(lowering, LOWERING)
    native_ref = read_artifact(lowering_ref)['native']
    receipt = verify_sparse_lowering(recipe_ref, lowering_ref, native_ref)
    provider = read_artifact(native_ref)
    blas_path = Path(provider['blas']['uri'])
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix='.scar-native-export-', dir=out.parent) as temporary:
        pending = Path(temporary)
        provider_ref = build_native_provider_bundle(provider['library'], pending / 'provider',
            provider_roots=(blas_path.parent,), system_sonames=_SYSTEM_LIBRARIES)
        bundle = validate_native_provider_bundle(provider_ref)
        blas = [row for row in bundle['members'] if row['origin'] == str(blas_path)]
        if len(blas) != 1 or blas[0]['sha256'] != provider['blas']['sha256']:
            raise ValueError('the export must use the original pinned numeric provider bytes')
        root, = [row for row in bundle['members'] if row['path'] == bundle['root']]
        if root['elf']['machine'] != 'Advanced Micro Devices X86-64':
            raise ValueError('the sparse export currently covers the qualified Linux x86-64 provider')
        (pending / _HEADER).write_bytes(_render(bundle))
        artifact = dict(schema=SCHEMA, schema_version=1, abi=_ABI, source_recipe_id=receipt['recipe_id'],
            build_provenance=dict(recipe=recipe_ref, lowering=lowering_ref, native=native_ref),
            provider=_local_artifact(pending, _PROVIDER_MANIFEST, PROVIDER),
            header=_local_artifact(pending, _HEADER, 'scar.cpp-header'),
            numeric_library_unchanged=root['sha256']==provider['library']['sha256'],
            original_python_entry_compatibility='separate existing entry and fallback',
            execution_cost='unmeasured for exported boundary', deployment_authorized=False)
        (pending / 'artifact.json').write_text(json.dumps(artifact, sort_keys=True, indent=2) + '\n')
        check_sparse_export(pending)
        if out.exists():
            raise FileExistsError(out)
        pending.rename(out)
    return artifact_reference(out / 'artifact.json', SCHEMA)


def check_sparse_export(path):
    """Verify an export after relocation without its original Python or library paths."""
    root = Path(path).resolve(strict=True)
    manifest = read_artifact(artifact_reference(root / 'artifact.json', SCHEMA))
    if (manifest.get('schema_version') != 1 or manifest['abi'] != _ABI
            or manifest['provider'] != _local_artifact(root, _PROVIDER_MANIFEST, PROVIDER)
            or manifest['header'] != _local_artifact(root, _HEADER, 'scar.cpp-header')
            or manifest['deployment_authorized'] is not False):
        raise ValueError('native export ABI or local artifact differs')
    bundle = validate_native_provider_bundle({**manifest['provider'], 'uri': str(root / _PROVIDER_MANIFEST)})
    if (set(bundle['system_allowlist']) != set(_SYSTEM_LIBRARIES)
            or sha256(_render(bundle)).hexdigest() != manifest['header']['sha256']):
        raise ValueError('native export loader does not match the ordered dependency closure')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe')
    parser.add_argument('--lowering')
    parser.add_argument('--out')
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    inputs = (args.recipe, args.lowering, args.out)
    if args.check and not any(inputs):
        report = check_sparse_export(args.check)
        print(json.dumps(dict(verified=True, recipe_id=report['source_recipe_id'])))
    elif not args.check and all(inputs):
        print(json.dumps(export_sparse_native(*inputs)))
    else:
        parser.error('choose --check BUNDLE or supply --recipe, --lowering and --out')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

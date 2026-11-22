"""Build a C++20 COO ingress using the pinned upstream sparse tool routines."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import shlex
from struct import pack
import tarfile

from scar.codegen.native_project import (
    build_native_project, open_native_package, render_cpp_template, stage_native_package, write_native_project,
)
from scar.codegen.native_sparse_source import SOURCE_STREAM
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.native-coo-ingress'
OBSERVATION = 'scar.coo-ingress-observation'
ARCHIVE_SHA256 = '52c4b7422442aba924d03ad4019852b08a92e64ea187b933135687bfe2747307'
_ROOT = 'scipy-1.18.1'
_HEADERS = ('coo.h', 'csr.h', 'dense.h', 'util.h')
_TARGET = 'scar_coo_ingress'
_FLAGS = ('-O2', '-fno-fast-math', '-ffp-contract=off')


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def _binding(observation):
    stream = read_artifact(observation['stream'])
    graph = read_artifact(observation['entry_calls'])
    scope = read_artifact(observation['scope'])
    entry = observation['entry']
    _require(observation['schema_version'] == 1 and stream['schema'] == SOURCE_STREAM
        and graph['schema'] == 'scar.bounded-entry-calls' and scope['entry']['source'] == entry
        and read_artifact(graph['program'])['entry'] == entry['uri']
        and observation['original_entry_effects_closed'] is False,
        'expected source-bound matrix and load assembly before canonical CSR ingress')
    read_artifact(entry, json_document=False)
    root = observation['original_code']['entry']
    providers = [value for key, value in observation['original_code'].items() if key != 'entry']
    _require(len(providers) == 2 and root['key'] == stream['source']['body_code_key']
        == scope['source_body']['code_key'], 'observed assembly belongs to another solver body')
    for provider in providers:
        label, index = provider['observation'].rsplit('/', 1)
        _require(label == '#/observed_targets' and index.isdecimal(), 'expected a retained producer observation')
        row = graph['observed_targets'][int(index)]
        _require(row['entry_scope_reachable'] and row['source_body'] is not None
            and {key: value for key, value in provider.items() if key != 'observation'}
                == row['observation']['target_code'],
            'COO operand evidence differs from its original retained producer')

    fields = observation['fields']

    def count(name, dtype):
        field = fields[name]
        size, = field['shape']
        _require(type(size) is int and 0 <= size < 2**31 - 1 and field['dtype'] == dtype
            and field['nbytes'] == size * int(dtype[-1]), 'COO operands require the declared contiguous wire types')
        return size

    matrix = count('coo-matrix-values', '<f8')
    rhs = count('coo-rhs-values', '<f8')
    boundary = count('boundary', '<i4')
    header = stream['header']
    n, csr, rounds = (header[key] for key in ('size', 'nnz', 'count'))
    _require(all(type(value) is int and 0 < value < 2**31 - 1 for value in (n, csr, rounds))
        and 0 < csr <= matrix and boundary <= n
        and count('coo-matrix-rows', '<i4') == count('coo-matrix-cols', '<i4') == matrix
        and count('coo-rhs-rows', '<i4') == rhs
        and count('csr-indices', '<i4') == count('csr-data', '<f8') == csr
        and count('csr-indptr', '<i4') == n + 1 and count('dense-rhs', '<f8') == n
        and header['format'] == '<8s32s4I' and stream['input']['magic'] == 'SCARPSR1',
        'COO materializations do not fit the qualified solver input geometry')
    canonical = pack(header['format'], b'SCARPSR1', bytes.fromhex(header['recipe_digest_hex']),
        n, csr, rounds, header['flags'])
    ingress = pack('<8s3I', b'SCARCOO1', matrix, rhs, boundary) + canonical
    original = read_artifact(observation['output_reference'], json_document=False)
    source = read_artifact(observation['input'], json_document=False)

    def check_payload(payload, prefix, names):
        _require(payload.startswith(prefix), 'retained operand header differs from the declared wire profile')
        offset = len(prefix)
        for name in names:
            if name is None:
                _require(payload[offset:offset+4] == pack('<I', boundary), 'retained boundary count changed')
                offset += 4
                continue
            field = fields[name]
            end = offset + field['nbytes']
            _require(end <= len(payload) and sha256(memoryview(payload)[offset:end]).hexdigest() == field['sha256'],
                f'retained operand bytes differ for {name}')
            offset = end
        _require(offset == len(payload), 'retained operand stream has trailing data')

    check_payload(source, ingress,
        ('coo-matrix-rows', 'coo-matrix-cols', 'coo-matrix-values', 'coo-rhs-rows', 'coo-rhs-values', 'boundary'))
    check_payload(original, canonical, ('csr-indices', 'csr-indptr', 'csr-data', 'dense-rhs', None, 'boundary'))
    _require(len(original) == stream['bytes']['input_fixed'] + 4*boundary,
        'retained canonical CSR length differs from the selected solver input')
    return dict(DIMENSION=n, CSR_COUNT=csr, MATRIX_COUNT=matrix, RHS_COUNT=rhs, BOUNDARY_COUNT=boundary,
        INGRESS=', '.join(str(value) for value in ingress))


def _upstream(source, output):
    _require(source['sha256'] == ARCHIVE_SHA256, 'expected the exact SciPy 1.18.1 source distribution')
    destination = output / 'vendor'
    destination.mkdir()
    with tarfile.open(fileobj=BytesIO(read_artifact(source, json_document=False)), mode='r:gz') as archive:
        for name in (*_HEADERS, 'LICENSE.txt'):
            folder = 'scipy/sparse/sparsetools/' if name in _HEADERS else ''
            with archive.extractfile(f'{_ROOT}/{folder}{name}') as stream:
                (destination / name).write_bytes(stream.read())


def export(observation_path, source_path, out):
    ref = artifact_reference(observation_path, OBSERVATION)
    observation = read_artifact(ref)
    source = artifact_reference(source_path, 'scar.upstream-source-archive')
    values = _binding(observation)
    out = Path(out).absolute()
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    with stage_native_package(out) as root:
        _upstream(source, root)
        template = Path(__file__).with_name('native_coo_ingress_v1.cpp.in')
        (root / 'main.cpp').write_bytes(render_cpp_template(template, {f'@{key}@': value for key, value in values.items()}))
        write_native_project(root, _TARGET, sources=('main.cpp',), include_dirs=('vendor',), compile_options=_FLAGS)
        executable, dependencies = build_native_project(root, _TARGET, compiler)
        members = {}
        for path in (executable, root/'main.cpp', root/'CMakeLists.txt', *sorted((root/'vendor').iterdir())):
            name = str(path.relative_to(root))
            members[name] = {**artifact_reference(path, 'scar.native-coo-member'), 'uri': name}
        result = dict(schema=SCHEMA, schema_version=1, observation=ref, upstream_source=source,
            stream=observation['stream'], input_sample=observation['input'],
            canonical_sample=observation['output_reference'],
            provider_routines=['coo_tocsr', 'csr_has_sorted_indices',
                'csr_sort_indices', 'csr_sum_duplicates', 'coo_todense'],
            members=members, dependencies=dependencies,
            compiler=dict(command=compiler, options=list(_FLAGS)), runtime='CXX20',
            boundary='CONTIGUOUS_COO_MATRIX_AND_LOAD_PLUS_SELECTED_ROWS_TO_CANONICAL_CSR',
            source_effects_discharged=False, deployment_authorized=False)
        (root/'artifact.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return artifact_reference(out/'artifact.json', SCHEMA)


def read_package(package):
    root = Path(package).resolve(strict=True)
    reference, result = open_native_package(root, SCHEMA)
    members = ('bin/' + _TARGET, 'main.cpp', 'CMakeLists.txt',
        *(f'vendor/{name}' for name in (*_HEADERS, 'LICENSE.txt')))
    _require(result['schema_version'] == 1 and result['runtime'] == 'CXX20'
        and result['upstream_source']['sha256'] == ARCHIVE_SHA256
        and result['source_effects_discharged'] is False and result['deployment_authorized'] is False
        and set(result['members']) == set(members), 'expected the bounded C++20 COO ingress package')
    for name in members:
        path = root / name
        ref = result['members'][name]
        _require(path.resolve(strict=True) == path and ref['uri'] == name,
            'COO package member differs from the relative layout')
        read_artifact({**ref, 'uri': str(path)}, json_document=False)
    _require(os.access(root / members[0], os.X_OK), 'COO ingress member is not executable')
    return reference, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation', type=Path, required=True)
    parser.add_argument('--scipy-source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(export(args.observation, args.scipy_source, args.out)))


if __name__ == '__main__':
    main()

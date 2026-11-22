"""Specialize the qualified ordered sparse C++ provider to bounded runtime counts."""

from __future__ import annotations

import argparse
import json
from math import isfinite
import os
from pathlib import Path
import shlex
from shutil import copytree
import struct
import subprocess
from tempfile import TemporaryDirectory

from scar.analysis.replay_transaction import replay_transaction
from scar.codegen import native_ordered_profile as fixed, native_ordered_sparse_export as ordered
from scar.codegen.native_project import build_native_project, write_native_project
from scar.codegen.native_sparse_source import render_source_scale
from scar.codegen.scalar_standalone import _dependencies_for
from scar.codegen.sparse_native_export import check_sparse_export
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.native-parameterized-sparse-solver'
STREAM = 'scar.native-parameterized-sparse-digest-stream'
MAGIC = 'SCARPSP1'
_HEADER = struct.Struct('<8s32s4I')
_COUNT_OFFSET = struct.calcsize('<8s32s2I')
_MAX_I64_EXACT_BINARY64 = 2**53
_SRC = ordered._STREAM_MAIN
_BIN = ordered._STREAM_EXECUTABLE
_PROTOCOL = ordered._STREAM_PLAN
_FLAGS = ordered._compile_flags(ordered.DIGEST_STREAM)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _reference(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _archive(path):
    path = Path(path).resolve(strict=True)
    root = path if path.is_dir() else path.parent
    _require(path == root or path == root / 'artifact.json', 'expected a fixed-count solver directory or manifest')
    document = fixed.check_profile_solver(root)
    ref = _reference(root / 'artifact.json', fixed.SCHEMA)
    plan = read_artifact(document['profile_plan'])
    _, _, _, publication, protocol = fixed._arc(plan)
    resources = plan['resources']
    _require(document['stream']['protocol'] == ordered._local(root, _PROTOCOL, ordered.DIGEST_STREAM)
             and plan['width'] == len(resources['placement']['worker_cpu_ids']) == resources['dispatch']['window']
             and protocol['input']['framing'] == 'REQUIRE_EOF_BEFORE_ANY_DISPATCH',
             'runtime count requires a bounded checked FIFO worker team and complete CSR before dispatch')
    return root, ref, document, resources, publication, protocol


def _interval(tree, maximum):
    """Prove integer exactness and nonzero divisors throughout the count domain."""
    op, left, *right = tree
    if not right and op == 'input' and type(left) is int and left in (0, 1):
        return (0, maximum - 1) if left == 0 else (1, maximum)
    if not right and op == 'constant':
        if type(left) is int and abs(left) <= _MAX_I64_EXACT_BINARY64:
            return left, left
        _require(type(left) is float and isfinite(left), 'unsupported runtime count source constant')
        return None
    _require(op in ('Add', 'Sub', 'Mult', 'Div') and len(right) == 1, 'unresolved runtime count scale expression')
    a, b = _interval(left, maximum), _interval(right[0], maximum)
    if op == 'Div':
        _require(b is not None and not b[0] <= 0 <= b[1], 'runtime count domain does not prove an exact nonzero divisor')
        return None
    if a is None or b is None:
        return None
    if op == 'Add':
        bounds = a[0] + b[0], a[1] + b[1]
    elif op == 'Sub':
        bounds = a[0] - b[1], a[1] - b[0]
    else:
        terms = [x * y for x in a for y in b]
        bounds = min(terms), max(terms)
    _require(max(map(abs, bounds)) <= _MAX_I64_EXACT_BINARY64,
             'runtime range cannot establish exact integer conversion to binary64')
    return bounds


def _projection(archive, maximum_count):
    _, reference, manifest, resources, _, previous = archive
    _require(type(maximum_count) is int and 1 <= maximum_count < 2**32,
             'maximum runtime count must fit the positive canonical uint32 port')
    tree = previous['source']['expression']
    _interval(tree, maximum_count)
    count = dict(minimum=1, maximum=maximum_count)
    old_header = previous['header']
    identity = dict(schema=STREAM, provider_package_sha256=reference['sha256'], recipe_id=manifest['recipe_id'],
                    input_magic=MAGIC, size=old_header['size'], nnz=old_header['nnz'], count=count, flags=old_header['flags'])
    return dict(schema=STREAM, schema_version=1, anchor=reference, anchor_recipe_id=manifest['recipe_id'],
        input={**previous['input'], 'magic': MAGIC},
        header={**old_header, 'count': count, 'count_offset': _COUNT_OFFSET,
                'recipe_digest_hex': semantic_sha256(identity), 'identity': 'PARAMETERIZED_PHYSICAL_PROTOCOL'},
        source=previous['source'], output=previous['output'], width=manifest['width'], window=resources['dispatch']['window'],
        placement=resources['placement'], memory=resources['memory'],
        allocation='FIXED_WORKER_TEAM_AND_WINDOW_INDEPENDENT_OF_COUNT',
        scale=dict(integer_range='PROVEN_EXACT_IN_ADMITTED_COUNT_INTERVAL',
                   floating_range='CHECK_EACH_REQUEST_BEFORE_PROVIDER', input_relation='0 <= sequence < count'),
        old_profile_timings_transferred=False, deployment_authorized=False)


@replay_transaction()
def protocol(anchor, *, maximum_count):
    """Read the typed header and resource bounds without building or running a solver."""
    return _projection(_archive(anchor), maximum_count)


def pack_header(document, count):
    header = document['header']
    interval = header['count']
    _require(document['schema'] == STREAM and header['format'] == _HEADER.format
             and document['input']['magic'] == MAGIC and type(count) is int
             and interval['minimum'] <= count <= interval['maximum'], 'invalid parameterized CSR header count or protocol')
    return _HEADER.pack(MAGIC.encode('ascii'), bytes.fromhex(header['recipe_digest_hex']),
                        header['size'], header['nnz'], count, header['flags'])


def _replace_once(text, old, new):
    _require(text.count(old) == 1, 'checked solver source does not expose exactly one runtime-count specialization site')
    return text.replace(old, new, 1)


def _render(archive, document):
    root, _, parent, _, _, previous = archive
    source_ref = parent['stream']['source']
    text = read_artifact({**source_ref, 'uri': str(root / source_ref['uri'])}, json_document=False).decode('utf8')
    header = document['header']
    count = header['count']['maximum']
    old_expr = render_source_scale(previous['source']['expression'], previous['header']['count'])
    new_expr = render_source_scale(document['source']['expression'], count).replace('sparse::count', 'request_count')
    text = _replace_once(text, previous['header']['recipe_digest_hex'], header['recipe_digest_hex'])
    text = _replace_once(text, '    put_unsigned(fields.subspan(48, 4), sparse::count);\n', '')
    text = _replace_once(text, f'constexpr auto input_header = header("{previous["input"]["magic"]}");',
                          f'constexpr auto input_header = header("{MAGIC}");\n'
                          f'constexpr std::uint64_t maximum_request_count = {count};')
    text = _replace_once(text, 'constexpr auto output_header = header("SCAROUT1");\n', '')
    text = _replace_once(text, ' && sparse::count <= UINT32_MAX', '')
    staged = _replace_once(text, 'void require_record(std::uint64_t sequence, Record kind) {', '@SCAR_OLD_CONSUMER@')
    start, legacy = staged.split('@SCAR_OLD_CONSUMER@', 1)
    end = 'using ScaleCopy = '
    _require(legacy.count(end) == 1 and 'int consume_binary_results()' in legacy,
             'runtime producer cannot identify its distinct old binary-consumer entry')
    text = start + end + legacy.split(end, 1)[1]
    for original, replacement in (
        ('void fill_source_scale(std::uint64_t sequence, std::span<std::byte> request) {',
         'void fill_source_scale(std::uint64_t sequence, std::uint64_t request_count, std::span<std::byte> request) {\n'
         '    (void)request_count;'),
        (f'    const double scale = {old_expr};', f'    const double scale = {new_expr};'),
        ('int run_requests(std::span<const int> indices, std::span<const int> offsets, DetachedInput& input) {',
         'int run_requests(std::span<const int> indices, std::span<const int> offsets, DetachedInput& input,\n'
         '                 std::uint64_t request_count) {'),
        ('const auto fill = [](std::uint64_t sequence, std::span<std::byte> request)',
         'const auto fill = [request_count](std::uint64_t sequence, std::span<std::byte> request)'),
        ('fill_source_scale(sequence, request);', 'fill_source_scale(sequence, request_count, request);'),
        ('        if (sparse::count == 0) {\n            require_eof();\n        }\n', ''),
        ('ordered::dispatch(config, sparse::count, factory, fill, retire);',
         'ordered::dispatch(config, request_count, factory, fill, retire);'),
        ('published != sparse::count', 'published != request_count'),
        ('int main(int argc, char** argv) {\n'
         '    if (argc != 1 && (argc != 2 || std::string_view(argv[1]) != "--consume-records")) {\n'
         '        std::fputs("ordered sparse digest accepts binary stdin and optional --consume-records\\n", stderr);',
         'int main(int argc, char**) {\n'
         '    if (argc != 1) {\n'
         '        std::fputs("parameterized sparse digest accepts canonical binary stdin\\n", stderr);'),
        ('        if (argc == 2) {\n            return consume_binary_results();\n        }\n', ''),
        ('        if (observed != input_header) {\n'
         '            throw InputFailure{"detached CSR header differs from the compiled recipe"};\n        }',
         '        auto count_field = std::span(observed).subspan<48, 4>();\n'
         '        const auto request_count = get_unsigned(count_field);\n'
         '        put_unsigned(count_field, 0);\n'
         '        if (observed != input_header || request_count == 0 || request_count > maximum_request_count) {\n'
         '            throw InputFailure{"parameterized CSR header or runtime count leaves the admitted protocol"};\n        }'),
        ('return run_requests(indices, offsets, input);', 'return run_requests(indices, offsets, input, request_count);'),
    ):
        text = _replace_once(text, original, replacement)
    _require('sparse::count' not in text and '--consume-records' not in text,
             'parameterized solver text still imports fixed-count behavior')
    return text.encode('utf8')


def _project(root):
    write_native_project(root, _BIN, sources=(_SRC,), include_dirs=('.',), compile_options=_FLAGS[1:-1],
                         link_options=(_FLAGS[-1],), libraries=('dl',), install_dirs=(('numeric', 'bin'), ('sha256', 'bin')))


def _wire(document):
    return (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf8')


def _check_outputs(root, archive, document, manifest):
    local = ordered._local
    _, anchor, parent, resources, publication, _ = archive
    numeric = check_sparse_export(root / 'numeric')
    crypto = ordered._bundled_crypto(root, parent['sha256_provider'], publication)
    _require(manifest['schema_version'] == 1 and manifest['anchor'] == anchor
             and manifest['width'] == document['width'] and manifest['window'] == document['window']
             and manifest['maximum_count'] == document['header']['count']['maximum']
             and manifest['deployment_authorized'] is False and manifest['python_runtime_required'] is False
             and manifest['numeric'] == parent['numeric'] == local(root, 'numeric/artifact.json', numeric['schema'])
             and manifest['sha256_provider'] == parent['sha256_provider'] and manifest['crypto_member'] == crypto
             and manifest['protocol'] == local(root, _PROTOCOL, STREAM)
             and (root / _PROTOCOL).read_bytes() == _wire(document)
             and manifest['headers'] == parent['headers'] == [local(root, name, 'scar.cpp-header') for name in fixed._MEMBERS]
             and (root / ordered._ORDERED_HEADER).read_bytes() == ordered._render(resources, parent['blas_member']),
             'dynamic sparse provider differs from its checked numeric, source or resource contracts')
    loaded = manifest['stream']
    _require(loaded['protocol'] == manifest['protocol'] and loaded['source'] == local(root, _SRC, 'scar.cpp-source')
             and loaded['executable'] == local(root, _BIN, 'scar.native-executable-file')
             and (root / _SRC).read_bytes() == _render(archive, document) and os.access(root / _BIN, os.X_OK)
             and loaded['options'] == list(_FLAGS) and loaded['cmake'] == local(root, 'CMakeLists.txt', 'scar.cmake-project')
             and loaded['elf_dependencies'] == _dependencies_for(root / _BIN), 'dynamic sparse solver source or ELF differs')
    with TemporaryDirectory(prefix='scar-dynamic-sparse-replay-') as temporary:
        _project(Path(temporary))
        _require((Path(temporary) / 'CMakeLists.txt').read_bytes() == (root / 'CMakeLists.txt').read_bytes(),
                 'dynamic sparse solver CMake project differs')


@replay_transaction()
def export(anchor, out, *, maximum_count):
    archive = _archive(anchor)
    document = _projection(archive, maximum_count)
    source_root, anchor_ref, parent, _, publication, _ = archive
    out = Path(out).resolve()
    _require(not out.exists() and not out.is_relative_to(source_root), 'expected a fresh solver directory outside the original package')
    out.parent.mkdir(parents=True, exist_ok=True)
    local = ordered._local
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    _require(compiler, 'an available C++20 compiler is required')
    version = subprocess.check_output([*compiler, '--version'], text=True).splitlines()[0]
    with TemporaryDirectory(prefix='.scar-dynamic-sparse-', dir=out.parent) as temporary:
        root = Path(temporary) / 'package'
        root.mkdir()
        for name in ('numeric', 'sha256'):
            copytree(source_root / name, root / name)
        for ref in parent['headers']:
            (root / ref['uri']).write_bytes(read_artifact({**ref, 'uri': str(source_root / ref['uri'])}, json_document=False))
        (root / _PROTOCOL).write_bytes(_wire(document))
        (root / _SRC).write_bytes(_render(archive, document))
        _project(root)
        binary, _ = build_native_project(root, _BIN, compiler)
        binary = binary.rename(root / _BIN)
        protocol_ref = local(root, _PROTOCOL, STREAM)
        result = dict(schema=SCHEMA, schema_version=1, anchor=anchor_ref, protocol=protocol_ref,
            width=document['width'], window=document['window'], maximum_count=maximum_count,
            numeric=parent['numeric'], sha256_provider=parent['sha256_provider'], headers=parent['headers'],
            crypto_member=ordered._bundled_crypto(root, parent['sha256_provider'], publication),
            stream=dict(protocol=protocol_ref, source=local(root, _SRC, 'scar.cpp-source'),
                executable=local(root, _BIN, 'scar.native-executable-file'),
                cmake=local(root, 'CMakeLists.txt', 'scar.cmake-project'), options=list(_FLAGS),
                compiler=dict(command=compiler, version=version), elf_dependencies=_dependencies_for(binary)),
            deployment_authorized=False, python_runtime_required=False)
        (root / 'artifact.json').write_bytes(_wire(result))
        _check_outputs(root, archive, document, result)
        if out.exists():
            raise FileExistsError(out)
        root.rename(out)
    return _reference(out / 'artifact.json', SCHEMA)


@replay_transaction()
def check(path):
    path = Path(path).resolve(strict=True)
    if path.is_dir():
        path /= 'artifact.json'
    reference = _reference(path, SCHEMA)
    result = read_artifact(reference)
    archive = _archive(result['anchor']['uri'])
    _check_outputs(path.parent, archive, _projection(archive, result['maximum_count']), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchor', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    parser.add_argument('--maximum-count', type=int)
    args = parser.parse_args(argv)
    inputs = args.anchor, args.out, args.maximum_count
    if args.check and all(item is None for item in inputs):
        result = check(args.check)
    elif args.check is None and all(item is not None for item in inputs):
        result = export(args.anchor, args.out, maximum_count=args.maximum_count)
    else:
        parser.error('provide --anchor, --out and --maximum-count, or --check')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()

"""Qualify ordered weighted-form ATen composition against a source-captured COO wire."""

import argparse
import json
from math import isfinite, prod
import os
from pathlib import Path
import shlex
from shutil import copy2
from struct import Struct
import subprocess
from tempfile import TemporaryDirectory, TemporaryFile

from scar.codegen import native_basis_transport
from scar.codegen.native_project import build_native_project, write_native_project
from scar.codegen.native_provider_bundle import _content
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.native-basis-coo-export'
_WITNESS = 'scar.basis-assembly-probe'
_TEMPLATE = Path(__file__).with_name('native_basis_coo_v1.cpp.in')
_TARGET = 'scar_basis_coo'
_HEADER = Struct('<8s4I8s3I8s32s4I')
_LIBRARIES = ('libtorch_cpu.so', 'libc10.so', 'libgomp.so.1')
_ROLE_SPECS = (('grad', '<f8'), ('value', '<f8'), ('quadrature', '<f8'), ('dofs', '<i4'), ('constraints', '<i4'))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _local(root, path, schema):
    member = root / path
    _require(member.resolve(strict=True) == member, 'expected a regular local basis member: ' + path)
    return dict(uri=path, schema=schema, **_content(member))


def _inputs(witness, point_order, scalar):
    from hashlib import sha256

    _require(witness['schema_version'] == 1 and witness['exact_topology'] is True
        and type(scalar) is float and isfinite(scalar), 'ordered forms require finite binary64 and an admitted witness')
    source = read_artifact(witness['entry'], json_document=False)
    upstream = read_artifact(witness['upstream'])
    for caller in witness['observed_assembly'].values():
        matches = [row['observation']['target_code'] for row in upstream['observed_targets']
            if row['entry_scope_reachable'] and (row['observation']['target_code'] or {}).get('key') == caller['code_id']]
        _require(matches and all(match['filename'] == caller['path'] for match in matches),
            'assembly input CodeID is not reachable in the bound source graph')
    ingress = {**witness['input'], 'schema': 'scar.native-basis-array-wire'}
    raw = read_artifact(ingress, json_document=False)
    expected = read_artifact(witness['original_coo'], json_document=False)
    fields = witness['fields']
    (magic, basis, components, elements, points, coo, nmatrix, nload, rows, csr, _recipe,
        dimension, nnz, count, flags) = _HEADER.unpack_from(raw)
    _require(all(value > 0 for value in (basis, components, elements, points, dimension, nnz, count))
        and (magic, coo, csr, flags) == (b'SCARBAS1', b'SCARCOO1', b'SCARPSR1', 0)
        and tuple(sorted(point_order)) == tuple(range(points))
        and nmatrix == basis * basis * elements and nload == basis * elements and rows <= dimension
        and raw[24:_HEADER.size] == expected[:76], 'basis shapes/order do not agree with the source COO envelope')
    shapes = ((basis, components, elements, points), (basis, elements, points), (elements, points),
        (basis, elements), (rows,))
    offset = _HEADER.size
    for (role, dtype), shape in zip(_ROLE_SPECS, shapes, strict=True):
        field = fields[role]
        size = prod(shape) * int(dtype[-1])
        _require(field == dict(dtype=dtype, shape=list(shape), size_bytes=size,
            sha256=sha256(raw[offset:offset + size]).hexdigest()), 'source basis field differs from its observed bytes')
        offset += size
    _require(offset == len(raw), 'source basis ingress contains trailing or missing bytes')
    return ingress, expected, dict(input_header=list(raw[:_HEADER.size]), basis=basis, components=components,
        elements=elements, points=points, rows=rows, point_order=list(point_order), component_order=list(range(components)),
        value_scale=scalar.hex(), capture_source_sha256=sha256(source).hexdigest(),
        operations=['ATen view/expand', 'aten::mul', 'aten::zeros_like', 'aten::add', 'ATen contiguous'])


def _render(plan):
    values = {key.upper(): str(plan[key]) for key in ('basis', 'components', 'elements', 'points', 'rows', 'value_scale')}
    values.update({key.upper(): ', '.join(map(str, plan[key])) for key in ('point_order', 'component_order')})
    values['HEADER'] = ', '.join(map(str, plan['input_header']))
    values['HEADER_SIZE'] = str(len(plan['input_header']))
    values['VALUE_READER'] = ('const auto value = read_tensor({basis, points}, at::kDouble)'
        '.unsqueeze(1).expand({basis, elements, points})' if 'dense_input_header' in plan
        else 'const auto value = read_tensor({basis, elements, points}, at::kDouble)')
    coo_bytes = len(plan['dense_input_header']) - 24 if 'dense_input_header' in plan else 'header.size() - 24'
    values['COO_HEADER_WRITE'] = f'write_bytes(header.data() + 24, {coo_bytes})'
    source = _TEMPLATE.read_text()
    for marker, value in values.items():
        source = source.replace(f'@SCAR_{marker}@', value)
    return source.encode()


def _value_view_contract(torch):
    schema = torch._C._dispatch_find_schema_or_throw
    views = [schema(name, '').schema() for name in ('aten::unsqueeze', 'aten::expand')]
    multiply = schema('aten::mul', 'Scalar').schema()
    for view in views:
        aliases = [view.arguments[0].alias_info, view.returns[0].alias_info]
        _require(all(alias is not None and not alias.is_write for alias in aliases),
            'compact Basis value requires provider views without mutation')
    _require(all(port.alias_info is None for port in (*multiply.arguments, *multiply.returns)),
        'compact Basis value must feed an out-of-place multiplication')
    return dict(schemas=[*map(str, views), str(multiply)], owner='RECEIVER_PRIVATE_THROUGH_LAST_USE',
        read_only=True, escapes=False)


def export(witness_path, out, *, point_order, scalar, compact_proof=None):
    import torch

    witness_ref = artifact_reference(Path(witness_path).resolve(strict=True), _WITNESS)
    witness = read_artifact(witness_ref)
    _require((torch.__version__, torch.version.git_version) == (witness['provider']['torch'], witness['provider']['git'])
        and torch.version.cuda is None and torch.version.hip is None, 'a matching CPU-only ATen provider is required')
    ingress, expected, plan = _inputs(witness, point_order, scalar)
    dense_input = ingress
    compact = None
    if compact_proof is not None:
        view_contract = _value_view_contract(torch)
        proof_ref, compact = native_basis_transport.read_compact(compact_proof)
        legacy = read_artifact(compact['consumer'])
        _require(compact['dense_input'] == dense_input and legacy['witness'] == witness_ref
            and legacy['plan'] == plan and legacy['original_coo'] == witness['original_coo'],
            'compact Basis proof does not bind this exact original assembly recipe')
        ingress = compact['compact_input']
        header = native_basis_transport.project_header(bytes(plan['input_header']), compact['wire'])
        compact_bytes = read_artifact(ingress, json_document=False)
        _require(len(compact_bytes) == compact['wire']['compact_bytes'] and compact_bytes.startswith(header),
            'compact Basis sample differs from its admitted envelope')
        plan.update(dense_input_header=plan['input_header'], input_header=list(header))
    torch_root = Path(torch.__file__).resolve(strict=True).parent
    originals = [torch_root / 'lib' / member for member in _LIBRARIES]
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-weighted-forms-', dir=out.parent) as temporary:
        root = Path(temporary)
        library = root / 'lib'
        library.mkdir()
        for source in originals:
            copy2(source, library / source.name)
        (root / 'main.cpp').write_bytes(_render(plan))
        write_native_project(root, _TARGET, sources=('main.cpp',),
            include_dirs=(str(torch_root / 'include'),), libraries=tuple(map(str, originals[:2])),
            compile_options=('-O2', '-fno-fast-math', '-ffp-contract=off',
                f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}'),
            runtime_dirs=('$ORIGIN/../lib',), install_dirs=(('lib', '.'),))
        launcher, dependencies = build_native_project(root, _TARGET, shlex.split(os.environ.get('CXX', 'c++')),
            expected_libraries=tuple(library / item for item in _LIBRARIES), require_cpu_only=True)
        with Path(ingress['uri']).open('rb') as original, TemporaryFile() as generated:
            run = subprocess.run([launcher], stdin=original, stdout=generated, stderr=subprocess.PIPE,
                env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'OMP_NUM_THREADS': '1'}, timeout=20, check=True)
            generated.seek(0)
            _require(not run.stderr and generated.read(len(expected) + 1) == expected,
                'native operator composition does not exactly reproduce the source COO values and ordering')
        result = dict(schema=SCHEMA, schema_version=2 if compact is not None else 1, witness=witness_ref, plan=plan,
            admitted_input=ingress, original_coo=witness['original_coo'], exact_native_coo=True,
            evidence_scope='ONE_CAPTURED_SOURCE_BASIS_AND_ORDERED_CONTRIBUTIONS', full_python_entry=False,
            runtime='CPU_ONLY_ATEN', source=_local(root, 'main.cpp', 'scar.cpp-source'),
            executable=_local(root, 'bin/' + _TARGET, 'scar.native-executable-file'),
            libraries=[_local(root, 'lib/' + item, 'scar.elf-library') for item in _LIBRARIES],
            linker_members=sorted(dependencies))
        if compact is not None:
            result.update(dense_admitted_input=dense_input, compact_proof=proof_ref, wire=compact['wire'],
                proof_member=native_basis_transport.write_proof(root, proof_ref), value_view=view_contract)
        (root / 'artifact.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
        if out.exists():
            raise FileExistsError(out)
        root.rename(out)
    return artifact_reference(out / 'artifact.json', SCHEMA)


def check(package):
    """Check the detached executable, source and locally resolved native providers."""
    from scar.codegen.aten_provider import _dependencies
    from scar.runtime.native_cpu_linkage import qualify_cpu_dependencies

    root = Path(package).resolve(strict=True)
    doc = read_artifact({**_local(root, 'artifact.json', SCHEMA), 'uri': str(root / 'artifact.json')})
    executable = root / 'bin' / _TARGET
    version = doc['schema_version']
    _require(version in (1, 2) and ('dense_input_header' in doc['plan']) == (version == 2)
        and doc['runtime'] == 'CPU_ONLY_ATEN' and doc['exact_native_coo'] is True
        and doc['full_python_entry'] is False and (root / 'main.cpp').read_bytes() == _render(doc['plan'])
        and doc['source'] == _local(root, 'main.cpp', 'scar.cpp-source')
        and doc['executable'] == _local(root, 'bin/' + _TARGET, 'scar.native-executable-file')
        and doc['libraries'] == [_local(root, 'lib/' + item, 'scar.elf-library') for item in _LIBRARIES]
        and os.access(executable, os.X_OK), 'basis project differs from its encoded native recipe or local providers')
    if version == 2:
        proof = native_basis_transport.read_local_proof(root, doc)
        plan = doc['plan']
        view = doc['value_view']
        _require(proof['dense_input'] == doc['dense_admitted_input'] and proof['compact_input'] == doc['admitted_input']
            and view['read_only'] is True and view['escapes'] is False and view['owner'] == 'RECEIVER_PRIVATE_THROUGH_LAST_USE'
            and bytes(plan['input_header']) == native_basis_transport.project_header(bytes(plan['dense_input_header']), doc['wire']),
            'compact Basis project does not consume the exact proven representation')
    ldd = subprocess.check_output(['ldd', str(executable)], env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'}, text=True)
    dependencies = _dependencies(ldd)
    qualify_cpu_dependencies(dependencies)
    _require(sorted(dependencies) == doc['linker_members']
        and all(dependencies.get(item) == str(root / 'lib' / item) for item in _LIBRARIES),
        'basis project ELF resolves a different set of providers')
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--witness', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--point-order', type=int, nargs='+')
    parser.add_argument('--value-scale', type=float)
    parser.add_argument('--compact-proof', type=Path)
    args = parser.parse_args(argv)
    if args.check and args.out is None and args.point_order is None and args.value_scale is None and args.compact_proof is None:
        check(args.check)
        print(json.dumps(dict(verified=True)))
    elif args.witness and args.out and args.point_order is not None and args.value_scale is not None:
        print(json.dumps(export(args.witness, args.out, point_order=args.point_order, scalar=args.value_scale,
            compact_proof=args.compact_proof)))
    else:
        parser.error('choose --check PACKAGE or --witness PROBE --out PACKAGE --point-order LOCATIONS --value-scale SCALAR')


if __name__ == '__main__':
    main()

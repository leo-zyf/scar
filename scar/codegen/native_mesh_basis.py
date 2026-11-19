"""Compose native ATen operators for a checked affine nodal triangle recipe."""

import argparse
from hashlib import sha256
import json
from math import prod
import os
from pathlib import Path
import shlex
from shutil import copy2
from struct import Struct
import subprocess
from tempfile import TemporaryDirectory, TemporaryFile

from scar.codegen import native_basis_coo, native_basis_transport
from scar.codegen.aten_provider import _dependencies
from scar.codegen.native_project import build_native_project, write_native_project
from scar.codegen.native_provider_bundle import _content
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.native_cpu_linkage import qualify_cpu_dependencies


SCHEMA = 'scar.native-mesh-basis-export'
WIRE = 'scar.native-triangular-mesh-wire'
CAPTURE = 'scar.mesh-basis-source-capture'
_TARGET = 'scar_mesh_basis'
_LIBRARIES = ('libtorch_cpu.so', 'libc10.so', 'libgomp.so.1')
_TEMPLATE = Path(__file__).with_name('native_mesh_basis_v1.cpp.in')
_MESH = Struct('<8s3I')
_BASIS = Struct('<8s4I8s3I8s32s4I')


def _require(valid, reason):
    if not valid:
        raise ValueError(reason)


def _local(root, name, schema):
    member = root / name
    _require(member.resolve(strict=True) == member, 'expected a local mesh package member')
    return dict(uri=name, schema=schema, **_content(member))


def _plan(mesh, basis, fields):
    magic, vertices, cells, flags = _MESH.unpack_from(mesh)
    embedded = mesh[_MESH.size:_MESH.size + _BASIS.size]
    (basis_magic, nbasis, components, basis_cells, points, coo, nmatrix, nload, rows,
        csr, _, dimension, _nnz, _count, basis_flags) = _BASIS.unpack(embedded)
    _require((magic, flags, basis_magic, nbasis, components, coo, csr, basis_flags) ==
        (b'SCARMES1', 0, b'SCARBAS1', 3, 2, b'SCARCOO1', b'SCARPSR1', 0)
        and min(vertices, cells, points) > 0 and dimension == vertices and cells == basis_cells
        and nmatrix == 9 * cells and nload == 3 * cells and rows <= vertices and basis.startswith(embedded),
        'mesh does not fit the admitted affine triangle Basis contract')
    mesh_fields = dict(mesh_coordinates=['<f8', [2, vertices]], mesh_connectivity=['<i4', [3, cells]],
        reference_values=['<f8', [3, points]], reference_derivatives=['<f8', [3, 2, points]],
        quadrature_weights=['<f8', [points]], constraints=['<i4', [rows]])
    offset = _MESH.size + _BASIS.size
    for role, (dtype, shape) in mesh_fields.items():
        size = int(dtype[-1]) * prod(shape)
        expected = dict(dtype=dtype, shape=shape, nbytes=size, sha256=sha256(mesh[offset:offset + size]).hexdigest())
        _require(all(fields[role][item] == value for item, value in expected.items()), 'mesh wire field differs from original source: ' + role)
        offset += size
    _require(offset == len(mesh), 'mesh wire length differs from its typed ports')
    return dict(input_header=list(mesh[:_MESH.size + _BASIS.size]), vertices=vertices, cells=cells,
        points=points, constraints=rows, mesh_ports=mesh_fields,
        profile='AFFINE_NODAL_TRIANGLE_BINARY64_SOURCE_ORDER_WITH_EXPLICIT_REFERENCE_AND_CONSTRAINTS',
        operations=['ATen index_select', 'aten::sub', 'aten::mul', 'aten::neg', 'aten::div',
                    'aten::stack', 'aten::zeros_like', 'aten::add', 'aten::abs', 'ATen views and contiguous'])


def _render(plan):
    values = {role.upper(): str(plan[role]) for role in ('vertices', 'cells', 'points', 'constraints')}
    values['HEADER'] = ', '.join(map(str, plan['input_header']))
    values.update(OUTPUT_HEADER_DECLARATION='',
        VALUE_VIEW='const auto values = reference_value.unsqueeze(1).expand({3, cells, points})',
        OUTPUT_HEADER_WRITE='write_bytes(observed.data() + 20, observed.size() - 20)')
    if 'compact_output_header' in plan:
        header = plan['compact_output_header']
        payload = ', '.join(map(str, header))
        values.update(OUTPUT_HEADER_DECLARATION=f'constexpr std::array<unsigned char, {len(header)}> output_header{{{payload}}};\n',
            VALUE_VIEW='const auto& values = reference_value',
            OUTPUT_HEADER_WRITE='write_bytes(output_header.data(), output_header.size())')
    source = _TEMPLATE.read_text()
    for name, value in values.items():
        source = source.replace(f'@SCAR_{name}@', value)
    return source.encode()


def _same(ref, other):
    return all(ref[key] == other[key] for key in ('uri', 'sha256', 'size_bytes'))


def export(mesh, capture, basis_package, out, *, compact_proof=None):
    import torch

    basis_package = Path(basis_package).resolve(strict=True)
    downstream = native_basis_coo.check(basis_package)
    exact_basis = downstream['admitted_input']
    needs_compact = compact_proof is not None
    _require(downstream['schema_version'] == (2 if needs_compact else 1),
        'mesh emitter and downstream Basis consumer must use the same proven transport')
    dense_basis = downstream['dense_admitted_input'] if needs_compact else exact_basis
    upstream = artifact_reference(Path(mesh).resolve(strict=True), WIRE)
    capture_ref = artifact_reference(Path(capture).resolve(strict=True), CAPTURE)
    observed = read_artifact(capture_ref)
    _require(observed['schema_version'] == 1 and _same(observed['mesh_wire'], upstream)
        and _same(observed['expected_basis'], dense_basis), 'mesh observation does not bind the exact downstream operands')
    basis = read_artifact(dense_basis, json_document=False)
    plan = _plan(read_artifact(upstream, json_document=False), basis, observed['fields'])
    if needs_compact:
        proof_ref, compact = native_basis_transport.read_compact(compact_proof)
        legacy = read_artifact(compact['producer'])
        _require(downstream['compact_proof'] == proof_ref and compact['dense_input'] == dense_basis
            and compact['compact_input'] == exact_basis and compact['mesh_capture'] == capture_ref
            and legacy['capture'] == capture_ref and legacy['admitted_input'] == upstream and legacy['plan'] == plan,
            'compact Basis proof does not bind this exact original mesh emitter and downstream consumer')
        header = native_basis_transport.project_header(basis[:_BASIS.size], compact['wire'])
        basis = read_artifact(exact_basis, json_document=False)
        _require(len(basis) == compact['wire']['compact_bytes'] and basis.startswith(header),
            'compact Basis sample differs from its admitted envelope')
        plan['compact_output_header'] = list(header)
    _require(torch.version.cuda is None and torch.version.hip is None, 'mesh emitter requires a CPU-only ATen provider')
    torch_root = Path(torch.__file__).resolve(strict=True).parent
    pinned_torch = torch_root / 'lib'
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-mesh-basis-', dir=out.parent) as temporary:
        root = Path(temporary)
        lib = root / 'lib'
        lib.mkdir()
        for ref, name in zip(downstream['libraries'], _LIBRARIES, strict=True):
            fingerprint = _content(pinned_torch / name)
            _require(ref['uri'] == 'lib/' + name and all(ref[key] == value for key, value in fingerprint.items()),
                'build ATen libraries differ from the downstream ABI')
            copy2(basis_package / ref['uri'], lib / name)
        (root / 'main.cpp').write_bytes(_render(plan))
        write_native_project(root, _TARGET, sources=('main.cpp',),
            include_dirs=(str(torch_root / 'include'),), libraries=tuple(str(pinned_torch / name) for name in _LIBRARIES[:2]),
            compile_options=('-O2', '-fno-fast-math', '-ffp-contract=off',
                f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}'),
            runtime_dirs=('$ORIGIN/../lib',), install_dirs=(('lib', '.'),))
        native, dependencies = build_native_project(root, _TARGET, shlex.split(os.environ.get('CXX', 'c++')),
            expected_libraries=tuple(lib / name for name in _LIBRARIES), require_cpu_only=True)
        with Path(upstream['uri']).open('rb') as source, TemporaryFile() as output:
            run = subprocess.run([native], stdin=source, stdout=output, stderr=subprocess.PIPE,
                env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'OMP_NUM_THREADS': '1'}, timeout=20, check=True)
            output.seek(0)
            _require(not run.stderr and output.read(len(basis) + 1) == basis, 'native mesh route differs from original Basis bytes')
        result = dict(schema=SCHEMA, schema_version=2 if needs_compact else 1, capture=capture_ref, admitted_input=upstream,
            exact_basis=exact_basis, basis_package=artifact_reference(basis_package / 'artifact.json', native_basis_coo.SCHEMA),
            plan=plan, scope='CAPTURED_MESH_REFERENCE_AND_CONSTRAINTS_TO_BASIS', runtime='CPU_ONLY_ATEN',
            verified_basis_bytes=True, full_python_entry=False, source=_local(root, 'main.cpp', 'scar.cpp-source'),
            executable=_local(root, 'bin/' + _TARGET, 'scar.native-executable-file'),
            libraries=[_local(root, 'lib/' + name, 'scar.elf-library') for name in _LIBRARIES],
            linker_members=sorted(dependencies))
        if needs_compact:
            result.update(dense_exact_basis=dense_basis, compact_proof=proof_ref, wire=compact['wire'],
                proof_member=native_basis_transport.write_proof(root, proof_ref))
        (root / 'artifact.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
        if out.exists():
            raise FileExistsError(out)
        root.rename(out)
    return artifact_reference(out / 'artifact.json', SCHEMA)


def check(package):
    root = Path(package).resolve(strict=True)
    manifest = read_artifact({**_local(root, 'artifact.json', SCHEMA), 'uri': str(root / 'artifact.json')})
    binary = root / 'bin' / _TARGET
    version = manifest['schema_version']
    _require(version in (1, 2) and ('compact_output_header' in manifest['plan']) == (version == 2)
        and manifest['runtime'] == 'CPU_ONLY_ATEN'
        and manifest['full_python_entry'] is False and manifest['verified_basis_bytes'] is True
        and manifest['source'] == _local(root, 'main.cpp', 'scar.cpp-source')
        and (root / 'main.cpp').read_bytes() == _render(manifest['plan'])
        and manifest['executable'] == _local(root, 'bin/' + _TARGET, 'scar.native-executable-file')
        and manifest['libraries'] == [_local(root, 'lib/' + name, 'scar.elf-library') for name in _LIBRARIES]
        and os.access(binary, os.X_OK), 'packaged mesh executable differs from its artifact')
    if version == 2:
        proof = native_basis_transport.read_local_proof(root, manifest)
        plan = manifest['plan']
        dense_header = bytes(plan['input_header'][_MESH.size:])
        _require(proof['dense_input'] == manifest['dense_exact_basis'] and proof['compact_input'] == manifest['exact_basis']
            and proof['mesh_capture'] == manifest['capture']
            and bytes(plan['compact_output_header']) == native_basis_transport.project_header(dense_header, manifest['wire']),
            'compact mesh package does not produce the exact proven representation')
    listing = subprocess.check_output(['ldd', str(binary)], env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'}, text=True)
    dependencies = _dependencies(listing)
    qualify_cpu_dependencies(dependencies)
    _require(sorted(dependencies) == manifest['linker_members'] and all(dependencies.get(name) == str(root / 'lib' / name)
        for name in _LIBRARIES), 'packaged mesh executable does not resolve its exact local providers')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--mesh', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--basis-package', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--compact-proof', type=Path)
    args = parser.parse_args(argv)
    if args.check and not any((args.capture, args.basis_package, args.out, args.compact_proof)):
        check(args.check)
        print(json.dumps(dict(verified=True)))
    elif args.mesh and all((args.capture, args.basis_package, args.out)):
        print(json.dumps(export(args.mesh, args.capture, args.basis_package, args.out, compact_proof=args.compact_proof)))
    else:
        parser.error('choose --check PACKAGE or --mesh INPUT --capture CAPTURE --basis-package BASIS --out PACKAGE')


if __name__ == '__main__':
    main()

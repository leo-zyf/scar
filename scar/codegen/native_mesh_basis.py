"""Lower the checked triangle, mesh and Basis ports to CPU ATen C++20 projects."""

import argparse
from hashlib import sha256
import json
from math import isfinite, prod
from pathlib import Path
from struct import Struct
import subprocess
from tempfile import TemporaryFile

from scar.codegen.aten_provider import build_cpu_aten_package, verify_cpu_aten_package
from scar.codegen.native_project import open_native_package, render_cpp_template, stage_native_package
from scar.ir.execution_recipe import artifact_reference, open_artifact, read_artifact


TOPOLOGY_SCHEMA = 'scar.native-triangle-topology-export'
MESH_SCHEMA = 'scar.native-mesh-basis-export'
BASIS_SCHEMA = 'scar.native-basis-coo-export'
CAPTURE = 'scar.mesh-basis-source-capture'
WIRE = 'scar.native-triangular-mesh-wire'
_WITNESS = 'scar.basis-assembly-probe'
_COMPACT_SCHEMA = 'scar.native-compact-basis-proof'
_COMPACT_MEMBER = 'compact-proof.json'
_STAGES = {
    'topology': (TOPOLOGY_SCHEMA, 'scar_triangle_topology', 'native_triangle_topology_v1.cpp.in'),
    'mesh': (MESH_SCHEMA, 'scar_mesh_basis', 'native_mesh_basis_v1.cpp.in'),
    'basis': (BASIS_SCHEMA, 'scar_basis_coo', 'native_basis_coo_v1.cpp.in'),
}
_SEED = Struct('<8s4I')
_MESH = Struct('<8s3I')
_BASIS_HEADER = Struct('<8s4I')
_BASIS = Struct('<8s4I8s3I8s32s4I')
_COO = Struct('<8s3I')
_STRIDES = Struct('<3I')
_REFERENCES = ('reference_values', 'reference_derivatives', 'quadrature_weights')
_ROLE_SPECS = (('grad', '<f8'), ('value', '<f8'), ('quadrature', '<f8'), ('dofs', '<i4'), ('constraints', '<i4'))
_ENV = {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'OMP_NUM_THREADS': '1'}


def _require(valid, reason):
    if not valid:
        raise ValueError(reason)


def _same(ref, other):
    return all(ref[key] == other[key] for key in ('uri', 'sha256', 'size_bytes'))


def _compact_header(dense, wire):
    magic, basis, _, elements, points = _BASIS_HEADER.unpack_from(dense)
    strides = (points * 8, 0, 8)
    if (magic != b'SCARBAS1' or len(dense) != wire['basis_header_bytes'] or len(dense) != 100
            or wire['magic'] != 'SCARBAS2' or wire['value_strides_bytes'] != list(strides)
            or wire['stride_header_bytes'] != _STRIDES.size
            or wire['dense_bytes'] - wire['compact_bytes'] != basis * (elements - 1) * points * 8 - _STRIDES.size):
        raise ValueError('compact Basis descriptor does not fit the exact dense header')
    return b'SCARBAS2' + dense[8:] + _STRIDES.pack(*strides)


def _selected_site(graph, recorded):
    row, = (row for row in graph['observed_targets'] if row['entry_scope_reachable']
        and row['observation']['target_code'] == recorded['target_code']
        and row['observation']['caller_site'] == recorded['caller_site'])
    return row['observation']['target_code']['key']


def _triangle_input(capture):
    graph = read_artifact(capture['source_graph'])
    source = capture['entry']
    _require(read_artifact(graph['program'])['entry'] == source['uri'], 'triangle capture names another program')
    read_artifact(source, json_document=False)
    seed, mesh = (read_artifact(capture[key], json_document=False) for key in ('seed_wire', 'mesh_wire'))
    prefix = seed[_SEED.size:]
    magic, seed_vertices, seed_cells, depth, flags = _SEED.unpack_from(seed)
    mesh_magic, vertices, cells, mesh_flags = _MESH.unpack_from(mesh)
    basis_magic, functions, components, elements, points = _BASIS_HEADER.unpack_from(mesh, _MESH.size)
    coo_magic, matrix, load, rows = _COO.unpack_from(mesh, _MESH.size + _BASIS_HEADER.size)
    _require(capture['schema_version'] == 1 and prefix[:120] == mesh[:120]
        and (magic, mesh_magic, basis_magic, coo_magic) == (b'SCARTRI1', b'SCARMES1', b'SCARBAS1', b'SCARCOO1')
        and flags == mesh_flags == 0 and (functions, components) == (3, 2)
        and 0 < depth <= 15 and 0 < seed_cells <= cells == elements == seed_cells * 4**depth
        and 0 < seed_vertices <= vertices < 2**31 and 0 < rows <= vertices and points > 0
        and matrix == 9*cells and load == 3*cells
        and capture['topology_conditions'] == dict(affine=True, sorted_cells=True, uniform_levels=depth,
            has_named_boundaries=False, has_named_subdomains=False, distributed=False, facets=[[0,1],[1,2],[0,2]]),
        'expected an admitted uniform affine triangle and its ordered mesh envelope')
    _require(capture['original_basis_bytes_equal'] is True and all(capture['routes'].values())
        and capture['launch']['exit_code'] == 0 and capture['launch']['installed_bindings'] == 0,
        'expected the observed original mesh and consumer identity routes')
    phases = capture['phases']
    steps = capture['refinement_steps']
    identities = {role: _selected_site(graph, phases[role]) for role in ('refine', 'rows')}
    _require(len(steps) == depth and len({_selected_site(graph, row) for row in steps}) == 1
        and all(phases['refine']['wall_ns'][0] <= row['wall_ns'][0] < row['wall_ns'][1]
                <= phases['refine']['wall_ns'][1] for row in steps),
        'original refinement did not observe all uniform steps at one code site')
    identities['uniform'] = steps[0]['target_code']['key']

    specs = {
        'seed_coordinates': ('<f8', [2, seed_vertices]), 'seed_connectivity': ('<i4', [3, seed_cells]),
        'mesh_coordinates': ('<f8', [2, vertices]), 'mesh_connectivity': ('<i4', [3, cells]),
        'reference_values': ('<f8', [3, points]), 'reference_derivatives': ('<f8', [3, 2, points]),
        'quadrature_weights': ('<f8', [points]), 'constraints': ('<i4', [rows]),
    }

    def check_fields(payload, offset, names):
        for role in names:
            dtype, shape = specs[role]
            size = prod(shape) * int(dtype[-1])
            end = offset + size
            _require(end <= len(payload) and capture['fields'][role] == dict(dtype=dtype, shape=shape,
                nbytes=size, sha256=sha256(memoryview(payload)[offset:end]).hexdigest()),
                'captured triangle operand differs for ' + role)
            offset = end
        _require(offset == len(payload), 'trailing triangle payload data')

    check_fields(seed, 144, ('seed_coordinates', 'seed_connectivity', *_REFERENCES))
    check_fields(mesh, 120, ('mesh_coordinates', 'mesh_connectivity', *_REFERENCES, 'constraints'))
    return dict(seed_vertices=seed_vertices, seed_cells=seed_cells, depth=depth,
        vertices=vertices, cells=cells, points=points, rows=rows,
        original_code=identities, header=list(seed[:144])), mesh


def _mesh_plan(mesh, basis, fields):
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


def _basis_inputs(witness, point_order, scalar):
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
        dimension, nnz, count, flags) = _BASIS.unpack_from(raw)
    _require(all(value > 0 for value in (basis, components, elements, points, dimension, nnz, count))
        and (magic, coo, csr, flags) == (b'SCARBAS1', b'SCARCOO1', b'SCARPSR1', 0)
        and tuple(sorted(point_order)) == tuple(range(points))
        and nmatrix == basis * basis * elements and nload == basis * elements and rows <= dimension
        and raw[24:_BASIS.size] == expected[:76], 'basis shapes/order do not agree with the source COO envelope')
    shapes = ((basis, components, elements, points), (basis, elements, points), (elements, points),
        (basis, elements), (rows,))
    offset = _BASIS.size
    for (role, dtype), shape in zip(_ROLE_SPECS, shapes, strict=True):
        field = fields[role]
        size = prod(shape) * int(dtype[-1])
        _require(field == dict(dtype=dtype, shape=list(shape), size_bytes=size,
            sha256=sha256(raw[offset:offset + size]).hexdigest()), 'source basis field differs from its observed bytes')
        offset += size
    _require(offset == len(raw), 'source basis ingress contains trailing or missing bytes')
    return ingress, expected, dict(input_header=list(raw[:_BASIS.size]), basis=basis, components=components,
        elements=elements, points=points, rows=rows, point_order=list(point_order), component_order=list(range(components)),
        value_scale=scalar.hex(), capture_source_sha256=sha256(source).hexdigest(),
        operations=['ATen view/expand', 'aten::mul', 'aten::zeros_like', 'aten::add', 'ATen contiguous'])


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


def _read_compact(path):
    from scar.optimization.native_compact_basis import check

    reference = artifact_reference(Path(path).resolve(strict=True), _COMPACT_SCHEMA)
    value = check(reference['uri'])
    read_artifact(reference, json_document=False)
    return reference, value


def _read_local_compact(root, document):
    member = root / _COMPACT_MEMBER
    _require(member.resolve(strict=True) == member, 'expected a local compact Basis proof')
    reference, proof = open_artifact(member, _COMPACT_SCHEMA)
    reference['uri'] = _COMPACT_MEMBER
    _require(document['proof_member'] == reference
        and {**document['compact_proof'], 'uri': _COMPACT_MEMBER} == reference
        and proof['wire'] == document['wire'], 'packed Basis proof differs from its original qualified identity')
    return proof


def render(stage, plan):
    if stage == 'topology':
        values = {key.upper(): plan[key] for key in ('seed_vertices', 'seed_cells', 'depth', 'vertices', 'cells', 'points', 'rows')}
        header = plan['header']
    elif stage == 'mesh':
        values = {key.upper(): plan[key] for key in ('vertices', 'cells', 'points', 'constraints')}
        header = plan['input_header']
        values.update(OUTPUT_HEADER_DECLARATION='',
            VALUE_VIEW='const auto values = reference_value.unsqueeze(1).expand({3, cells, points})',
            OUTPUT_HEADER_WRITE='write_bytes(observed.data() + 20, observed.size() - 20)')
        if 'compact_output_header' in plan:
            compact = plan['compact_output_header']
            payload = ', '.join(map(str, compact))
            values.update(
                OUTPUT_HEADER_DECLARATION=f'constexpr std::array<unsigned char, {len(compact)}> output_header{{{payload}}};\n',
                VALUE_VIEW='const auto& values = reference_value',
                OUTPUT_HEADER_WRITE='write_bytes(output_header.data(), output_header.size())')
    elif stage == 'basis':
        values = {key.upper(): plan[key] for key in ('basis', 'components', 'elements', 'points', 'rows', 'value_scale')}
        values.update({key.upper(): ', '.join(map(str, plan[key])) for key in ('point_order', 'component_order')})
        header = plan['input_header']
        values['HEADER_SIZE'] = len(header)
        values['VALUE_READER'] = ('const auto value = read_tensor({basis, points}, at::kDouble)'
            '.unsqueeze(1).expand({basis, elements, points})' if 'dense_input_header' in plan
            else 'const auto value = read_tensor({basis, elements, points}, at::kDouble)')
        coo_bytes = len(plan['dense_input_header']) - 24 if 'dense_input_header' in plan else 'header.size() - 24'
        values['COO_HEADER_WRITE'] = f'write_bytes(header.data() + 24, {coo_bytes})'
    else:
        raise ValueError('unknown native mesh stage: ' + stage)
    values['HEADER'] = ', '.join(map(str, header))
    template = Path(__file__).with_name(_STAGES[stage][2])
    return render_cpp_template(template, {f'@SCAR_{key}@': value for key, value in values.items()})


def _publish(stage, out, document, expected, torch, *, pins=None, library_source=None):
    out = Path(out).resolve()
    schema, target, _ = _STAGES[stage]
    with stage_native_package(out) as root:
        (root / 'main.cpp').write_bytes(render(stage, document['plan']))
        executable, linked = build_cpu_aten_package(root, target, torch, pins=pins, library_source=library_source)
        with Path(document['admitted_input']['uri']).open('rb') as source, TemporaryFile() as output:
            run = subprocess.run([executable], stdin=source, stdout=output, stderr=subprocess.PIPE,
                env=_ENV, timeout=20, check=True)
            output.seek(0)
            _require(not run.stderr and output.read(len(expected) + 1) == expected,
                'native stage differs from its admitted sample: ' + stage)
        if 'compact_proof' in document:
            proof = document['compact_proof']
            (root / _COMPACT_MEMBER).write_bytes(read_artifact(proof, json_document=False))
            document['proof_member'] = {**artifact_reference(root / _COMPACT_MEMBER, _COMPACT_SCHEMA), 'uri': _COMPACT_MEMBER}
        value = dict(schema=schema, **document, **linked)
        (root / 'artifact.json').write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return artifact_reference(out / 'artifact.json', schema)


def export_topology(capture_path, basis_package, out):
    import torch

    reference = artifact_reference(Path(capture_path).resolve(strict=True), CAPTURE)
    capture = read_artifact(reference)
    _, basis = read_package(basis_package, 'basis')
    _require(basis['admitted_input'] == capture['expected_basis'], 'expected a matching CPU-only basis provider and capture')
    plan, expected = _triangle_input(capture)
    return _publish('topology', out, dict(schema_version=1, capture=reference, admitted_input=capture['seed_wire'],
        exact_mesh=capture['mesh_wire'], plan=plan, original_entry_effects_closed=False, exact_mesh_bytes=True),
        expected, torch, pins=basis['libraries'])


def export_mesh(mesh, capture, basis_package, out, *, compact_proof=None):
    import torch

    basis_package = Path(basis_package).resolve(strict=True)
    basis_reference, downstream = read_package(basis_package, 'basis')
    exact_basis = downstream['admitted_input']
    compact = compact_proof is not None
    _require(downstream['schema_version'] == (2 if compact else 1),
        'mesh emitter and downstream Basis consumer must use the same proven transport')
    dense_basis = downstream['dense_admitted_input'] if compact else exact_basis
    upstream = artifact_reference(Path(mesh).resolve(strict=True), WIRE)
    capture_ref = artifact_reference(Path(capture).resolve(strict=True), CAPTURE)
    observed = read_artifact(capture_ref)
    _require(observed['schema_version'] == 1 and _same(observed['mesh_wire'], upstream)
        and _same(observed['expected_basis'], dense_basis), 'mesh observation does not bind the exact downstream operands')
    basis = read_artifact(dense_basis, json_document=False)
    plan = _mesh_plan(read_artifact(upstream, json_document=False), basis, observed['fields'])
    document = dict(schema_version=2 if compact else 1, capture=capture_ref, admitted_input=upstream,
        exact_basis=exact_basis, basis_package={**basis_reference, 'uri': str(basis_package / 'artifact.json')},
        plan=plan, scope='CAPTURED_MESH_REFERENCE_AND_CONSTRAINTS_TO_BASIS',
        verified_basis_bytes=True, full_python_entry=False)
    if compact:
        proof_ref, proof = _read_compact(compact_proof)
        legacy = read_artifact(proof['producer'])
        _require(downstream['compact_proof'] == proof_ref and proof['dense_input'] == dense_basis
            and proof['compact_input'] == exact_basis and proof['mesh_capture'] == capture_ref
            and legacy['capture'] == capture_ref and legacy['admitted_input'] == upstream and legacy['plan'] == plan,
            'compact Basis proof does not bind this exact original mesh emitter and downstream consumer')
        header = _compact_header(basis[:_BASIS.size], proof['wire'])
        basis = read_artifact(exact_basis, json_document=False)
        _require(len(basis) == proof['wire']['compact_bytes'] and basis.startswith(header),
            'compact Basis sample differs from its admitted envelope')
        plan['compact_output_header'] = list(header)
        document.update(dense_exact_basis=dense_basis, compact_proof=proof_ref, wire=proof['wire'])
    return _publish('mesh', out, document, basis, torch, pins=downstream['libraries'],
                    library_source=basis_package / 'lib')


def export_basis(witness_path, out, *, point_order, scalar, compact_proof=None):
    import torch

    witness_ref = artifact_reference(Path(witness_path).resolve(strict=True), _WITNESS)
    witness = read_artifact(witness_ref)
    _require((torch.__version__, torch.version.git_version) == (witness['provider']['torch'], witness['provider']['git']),
        'expected the exact ATen provider of the source assembly witness')
    ingress, expected, plan = _basis_inputs(witness, point_order, scalar)
    compact = compact_proof is not None
    document = dict(schema_version=2 if compact else 1, witness=witness_ref, plan=plan,
        admitted_input=ingress, original_coo=witness['original_coo'], exact_native_coo=True,
        evidence_scope='ONE_CAPTURED_SOURCE_BASIS_AND_ORDERED_CONTRIBUTIONS', full_python_entry=False)
    if compact:
        view = _value_view_contract(torch)
        proof_ref, proof = _read_compact(compact_proof)
        legacy = read_artifact(proof['consumer'])
        _require(proof['dense_input'] == ingress and legacy['witness'] == witness_ref
            and legacy['plan'] == plan and legacy['original_coo'] == witness['original_coo'],
            'compact Basis proof does not bind this exact original assembly recipe')
        dense_input, ingress = ingress, proof['compact_input']
        header = _compact_header(bytes(plan['input_header']), proof['wire'])
        compact_bytes = read_artifact(ingress, json_document=False)
        _require(len(compact_bytes) == proof['wire']['compact_bytes'] and compact_bytes.startswith(header),
            'compact Basis sample differs from its admitted envelope')
        plan.update(dense_input_header=plan['input_header'], input_header=list(header))
        document.update(admitted_input=ingress, dense_admitted_input=dense_input,
                        compact_proof=proof_ref, wire=proof['wire'], value_view=view)
    return _publish('basis', out, document, expected, torch)


def _check_compact(root, stage, document):
    proof = _read_local_compact(root, document)
    plan = document['plan']
    if stage == 'mesh':
        dense_header = bytes(plan['input_header'][_MESH.size:])
        _require(proof['dense_input'] == document['dense_exact_basis'] and proof['compact_input'] == document['exact_basis']
            and proof['mesh_capture'] == document['capture']
            and bytes(plan['compact_output_header']) == _compact_header(dense_header, document['wire']),
            'compact mesh package does not produce the exact proven representation')
    else:
        view = document['value_view']
        _require(proof['dense_input'] == document['dense_admitted_input']
            and proof['compact_input'] == document['admitted_input']
            and view['read_only'] is True and view['escapes'] is False and view['owner'] == 'RECEIVER_PRIVATE_THROUGH_LAST_USE'
            and bytes(plan['input_header']) == _compact_header(bytes(plan['dense_input_header']), document['wire']),
            'compact Basis project does not consume the exact proven representation')


def read_package(package, stage):
    root = Path(package).resolve(strict=True)
    schema, target, _ = _STAGES[stage]
    reference, document = open_native_package(root, schema)
    version, plan = document['schema_version'], document['plan']
    if stage == 'topology':
        valid = version == 1 and document['exact_mesh_bytes'] is True and document['original_entry_effects_closed'] is False
    elif stage == 'mesh':
        valid = (version in (1, 2) and ('compact_output_header' in plan) == (version == 2)
                 and document['verified_basis_bytes'] is True and document['full_python_entry'] is False)
    else:
        valid = (version in (1, 2) and ('dense_input_header' in plan) == (version == 2)
                 and document['exact_native_coo'] is True and document['full_python_entry'] is False)
    _require(valid, 'unsupported native mesh package: ' + stage)
    if version == 2:
        _check_compact(root, stage, document)
    verify_cpu_aten_package(root, target, document, render(stage, plan))
    return reference, document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage', required=True)
    check = commands.add_parser('check')
    check.add_argument('kind', choices=_STAGES)
    check.add_argument('package', type=Path)
    topology = commands.add_parser('topology')
    mesh = commands.add_parser('mesh')
    basis = commands.add_parser('basis')
    for command in (topology, mesh, basis):
        command.add_argument('--out', type=Path, required=True)
    for command in (topology, mesh):
        command.add_argument('--capture', type=Path, required=True)
        command.add_argument('--basis-package', type=Path, required=True)
    for command in (mesh, basis):
        command.add_argument('--compact-proof', type=Path)
    mesh.add_argument('--mesh', type=Path, required=True)
    basis.add_argument('--witness', type=Path, required=True)
    basis.add_argument('--point-order', type=int, nargs='+', required=True)
    basis.add_argument('--value-scale', type=float, required=True)
    args = parser.parse_args(argv)
    if args.stage == 'check':
        read_package(args.package, args.kind)
        result = dict(verified=True)
    elif args.stage == 'topology':
        result = export_topology(args.capture, args.basis_package, args.out)
    elif args.stage == 'mesh':
        result = export_mesh(args.mesh, args.capture, args.basis_package, args.out, compact_proof=args.compact_proof)
    else:
        result = export_basis(args.witness, args.out, point_order=args.point_order,
                              scalar=args.value_scale, compact_proof=args.compact_proof)
    print(json.dumps(result))


if __name__ == '__main__':
    main()

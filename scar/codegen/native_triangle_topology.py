"""Lower an admitted sorted triangle refinement and boundary route to CPU ATen."""

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
from tempfile import TemporaryDirectory

from scar.codegen import native_basis_coo
from scar.codegen.native_project import build_native_project, write_native_project
from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.native-triangle-topology-export'
CAPTURE = 'scar.mesh-basis-source-capture'
_TARGET = 'scar_triangle_topology'
_TEMPLATE = Path(__file__).with_name('native_triangle_topology_v1.cpp.in')
_SEED = Struct('<8s4I')
_MESH = Struct('<8s3I')
_BASIS = Struct('<8s4I')
_COO = Struct('<8s3I')
_REFERENCES = ('reference_values', 'reference_derivatives', 'quadrature_weights')
_LIBRARIES = native_basis_coo._LIBRARIES
_local = native_basis_coo._local


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def _selected_site(graph, recorded):
    row, = (row for row in graph['observed_targets'] if row['entry_scope_reachable']
        and row['observation']['target_code'] == recorded['target_code']
        and row['observation']['caller_site'] == recorded['caller_site'])
    return row['observation']['target_code']['key']


def _input(capture):
    graph = read_artifact(capture['source_graph'])
    source = capture['entry']
    _require(read_artifact(graph['program'])['entry'] == source['uri'], 'triangle capture names another program')
    read_artifact(source, json_document=False)
    seed, mesh = (read_artifact(capture[key], json_document=False) for key in ('seed_wire', 'mesh_wire'))
    prefix = seed[_SEED.size:]
    magic, seed_vertices, seed_cells, depth, flags = _SEED.unpack_from(seed)
    mesh_magic, vertices, cells, mesh_flags = _MESH.unpack_from(mesh)
    basis_magic, functions, components, elements, points = _BASIS.unpack_from(mesh, _MESH.size)
    coo_magic, matrix, load, rows = _COO.unpack_from(mesh, _MESH.size + _BASIS.size)
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


def _render(plan):
    values = {key.upper(): str(plan[key]) for key in ('seed_vertices', 'seed_cells',
        'depth', 'vertices', 'cells', 'points', 'rows')}
    values['HEADER'] = ', '.join(map(str, plan['header']))
    source = _TEMPLATE.read_text()
    for name, value in values.items():
        source = source.replace(f'@SCAR_{name}@', value)
    return source.encode()


def export(capture_path, provider, out):
    import torch

    reference = artifact_reference(Path(capture_path).resolve(strict=True), CAPTURE)
    capture = read_artifact(reference)
    basis = native_basis_coo.check(provider)
    _require(basis['admitted_input'] == capture['expected_basis'] and torch.version.cuda is None
        and torch.version.hip is None, 'matching CPU-only basis provider and capture required')
    plan, expected = _input(capture)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch_root = Path(torch.__file__).resolve(strict=True).parent
    with TemporaryDirectory(prefix='.scar-triangle-', dir=out.parent) as temporary:
        root = Path(temporary)
        library = root/'lib'
        library.mkdir()
        originals = [torch_root/'lib'/name for name in _LIBRARIES]
        for original, pin in zip(originals, basis['libraries'], strict=True):
            copy2(original, library / original.name)
            _require(_local(root, 'lib/'+original.name, 'scar.elf-library') == pin,
                'triangle refinement cannot use a different ATen provider')
        (root/'main.cpp').write_bytes(_render(plan))
        write_native_project(root, _TARGET, sources=('main.cpp',), include_dirs=(str(torch_root/'include'),),
            libraries=tuple(map(str, originals[:2])), runtime_dirs=('$ORIGIN/../lib',),
            compile_options=('-O2', '-fno-fast-math', '-ffp-contract=off',
                f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}'), install_dirs=(('lib', '.'),))
        launcher, deps = build_native_project(root, _TARGET, shlex.split(os.environ.get('CXX', 'c++')),
            expected_libraries=tuple(library / name for name in _LIBRARIES), require_cpu_only=True)
        with Path(capture['seed_wire']['uri']).open('rb') as data:
            run = subprocess.run([launcher], stdin=data, capture_output=True, timeout=20, check=True,
                env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'OMP_NUM_THREADS': '1'})
        _require(not run.stderr and run.stdout == expected, 'ATen refinement did not reproduce the captured mesh wire')
        receipt = dict(schema=SCHEMA, schema_version=1, runtime='CPU_ONLY_ATEN', capture=reference,
            source=_local(root, 'main.cpp', 'scar.cpp-source'),
            executable=_local(root, 'bin/'+_TARGET, 'scar.native-executable-file'),
            admitted_input=capture['seed_wire'], exact_mesh=capture['mesh_wire'], plan=plan,
            libraries=basis['libraries'], linker_members=sorted(deps),
            original_entry_effects_closed=False, exact_mesh_bytes=True)
        (root/'artifact.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+'\n')
        if out.exists():
            raise FileExistsError(out)
        root.rename(out)
    return artifact_reference(out/'artifact.json', SCHEMA)


def check(package):
    from scar.codegen.aten_provider import _dependencies
    from scar.runtime.native_cpu_linkage import qualify_cpu_dependencies

    root = Path(package).resolve(strict=True)
    doc = read_artifact(artifact_reference(root/'artifact.json', SCHEMA))
    binary = 'bin/'+_TARGET
    _require(doc['schema_version'] == 1 and doc['runtime'] == 'CPU_ONLY_ATEN' and doc['exact_mesh_bytes'] is True
        and doc['original_entry_effects_closed'] is False and (root/'main.cpp').read_bytes() == _render(doc['plan'])
        and doc['source'] == _local(root, 'main.cpp', 'scar.cpp-source')
        and doc['executable'] == _local(root, binary, 'scar.native-executable-file')
        and doc['libraries'] == [_local(root, 'lib/'+name, 'scar.elf-library') for name in _LIBRARIES]
        and os.access(root/binary, os.X_OK), 'triangle project differs from its sealed local parts')
    deps = _dependencies(subprocess.check_output(['ldd', str(root/binary)],
        env={'PATH':'/usr/bin:/bin', 'LC_ALL':'C'}, text=True))
    qualify_cpu_dependencies(deps)
    _require(sorted(deps) == doc['linker_members']
        and all(deps.get(name) == str(root/'lib'/name) for name in _LIBRARIES),
        'triangle ELF resolves a different CPU provider')
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--capture', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--basis-provider', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.check and args.basis_provider is None and args.out is None:
        check(args.check)
        print(json.dumps(dict(verified=True)))
    elif args.capture and args.basis_provider and args.out:
        print(json.dumps(export(args.capture, args.basis_provider, args.out)))
    else:
        parser.error('choose --check PACKAGE or --capture CAPTURE --basis-provider PACKAGE --out OUTPUT')


if __name__ == '__main__':
    main()

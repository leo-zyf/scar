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
    source = _TEMPLATE.read_text()
    for marker, value in values.items():
        source = source.replace(f'@SCAR_{marker}@', value)
    return source.encode()


def export(witness_path, out, *, point_order, scalar):
    import torch

    witness_ref = artifact_reference(Path(witness_path).resolve(strict=True), _WITNESS)
    witness = read_artifact(witness_ref)
    _require((torch.__version__, torch.version.git_version) == (witness['provider']['torch'], witness['provider']['git'])
        and torch.version.cuda is None and torch.version.hip is None, 'a matching CPU-only ATen provider is required')
    ingress, expected, plan = _inputs(witness, point_order, scalar)
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
        result = dict(schema=SCHEMA, schema_version=1, witness=witness_ref, plan=plan,
            admitted_input=ingress, original_coo=witness['original_coo'], exact_native_coo=True,
            evidence_scope='ONE_CAPTURED_SOURCE_BASIS_AND_ORDERED_CONTRIBUTIONS', full_python_entry=False,
            runtime='CPU_ONLY_ATEN', source=_local(root, 'main.cpp', 'scar.cpp-source'),
            executable=_local(root, 'bin/' + _TARGET, 'scar.native-executable-file'),
            libraries=[_local(root, 'lib/' + item, 'scar.elf-library') for item in _LIBRARIES],
            linker_members=sorted(dependencies))
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
    _require(doc['schema_version'] == 1 and doc['runtime'] == 'CPU_ONLY_ATEN' and doc['exact_native_coo'] is True
        and doc['full_python_entry'] is False and (root / 'main.cpp').read_bytes() == _render(doc['plan'])
        and doc['source'] == _local(root, 'main.cpp', 'scar.cpp-source')
        and doc['executable'] == _local(root, 'bin/' + _TARGET, 'scar.native-executable-file')
        and doc['libraries'] == [_local(root, 'lib/' + item, 'scar.elf-library') for item in _LIBRARIES]
        and os.access(executable, os.X_OK), 'basis project differs from its encoded native recipe or local providers')
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
    args = parser.parse_args(argv)
    if args.check and args.out is None and args.point_order is None and args.value_scale is None:
        check(args.check)
        print(json.dumps(dict(verified=True)))
    elif args.witness and args.out and args.point_order is not None and args.value_scale is not None:
        print(json.dumps(export(args.witness, args.out, point_order=args.point_order, scalar=args.value_scale)))
    else:
        parser.error('choose --check PACKAGE or --witness PROBE --out PACKAGE --point-order LOCATIONS --value-scale SCALAR')


if __name__ == '__main__':
    main()

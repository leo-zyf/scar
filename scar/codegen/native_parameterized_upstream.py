"""Rebind the count and solver identity in an admitted dense native upstream ABI."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
from shutil import copy2, copytree
from struct import Struct
import subprocess

from scar.analysis import source_triangle_seed
from scar.codegen import native_coo_ingress, native_mesh_basis, source_triangle_seed_wire
from scar.codegen.aten_provider import stage_cpu_aten_libraries
from scar.codegen.native_project import build_native_project, open_native_package, stage_native_package
from scar.codegen.native_sparse_source import SOURCE_STREAM
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.native-parameterized-upstream'
PROFILE = 'scar.native-upstream-wire-profile'
_CSR = Struct('<8s32s4I')
_NAMES = ('topology', 'mesh', 'basis', 'coo')
_INPUT_PORTS = ('admitted_input', 'admitted_input', 'admitted_input', 'input_sample')
_OUTPUT_PORTS = ('exact_mesh', 'exact_basis', 'original_coo', 'canonical_sample')
_WIRES = ('seed.bin', 'wires/mesh.bin', 'wires/basis.bin', 'wires/coo.bin', 'canonical.csr')
_SIZES = (144, 120, 100, 76, 56)
_SOURCE_INVARIANTS = ('entry', 'scope', 'source_root', 'sources', 'depth', 'fields', 'topology_conditions')
_ENV = {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'OMP_NUM_THREADS': '1'}
_HEADER_NAMES = re.compile(r'\b(header|input_header|observed|ingress)\b')
_COMMENT_OR_STRING = re.compile(r'//[^\n]*|"(?:\\.|[^"\\])*"')
_READ_FORMS = ('std::array<unsigned char, input_header.size()> {value}{initial};',
    'if (std::fread({value}.data(), 1, {value}.size(), stdin) != {value}.size() || {value} != input_header) {{')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _local(root, name, schema):
    relative = Path(name)
    _require(not relative.is_absolute() and '..' not in relative.parts and relative.as_posix() == name,
        'expected a canonical relative upstream member')
    member = root / relative
    _require(member.resolve(strict=True) == member, 'expected regular local native upstream member: ' + name)
    return {**artifact_reference(member, schema), 'uri': name}


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _actual(root, reference):
    _require(reference == _local(root, reference['uri'], reference['schema']), 'native upstream member differs from its receipt')
    return read_artifact({**reference, 'uri': str(root / reference['uri'])}, json_document=False)


def _rebind(header, count, recipe_id):
    _require(type(count) is int and 0 < count < 2**31 - 1 and type(recipe_id) is str
        and re.fullmatch(r'recipe-v1:[0-9a-f]{64}', recipe_id) is not None, 'expected a typed positive count and canonical recipe ID')
    magic, _, n, nnz, _, flags = _CSR.unpack_from(header, len(header) - _CSR.size)
    _require(magic == b'SCARPSR1' and 0 < n < 2**31 and nnz > 0 and flags == 0,
        'count rebinding requires the qualified canonical CSR envelope')
    return header[:-_CSR.size] + _CSR.pack(magic, bytes.fromhex(recipe_id.removeprefix('recipe-v1:')), n, nnz, count, flags)


def _metadata_only(source, stage, wire):
    """Check that all uses of the declared header flow through read/compare/write only."""
    encoded = ', '.join(map(str, wire))
    if stage == 'coo':
        expected = [f'constexpr std::array<unsigned char, 76> ingress = {{{encoded}}};',
            'std::array<unsigned char, ingress.size()> header{};', 'read_exact(header.data(), header.size());',
            'if (header != ingress) {',
            'write_exact(header.data() + coarse_header_bytes, header.size() - coarse_header_bytes);']
    else:
        value = 'observed' if stage == 'mesh' else 'header'
        offset = {'topology': 'seed_header_bytes', 'mesh': '20', 'basis': '24'}[stage]
        expected = [f'constexpr std::array<unsigned char, {len(wire)}> input_header{{{encoded}}};',
            *(form.format(value=value, initial='{}' if stage == 'topology' else '') for form in _READ_FORMS),
            f'write_bytes({value}.data() + {offset}, {value}.size() - {offset});']
    routes = [line.strip() for line in _COMMENT_OR_STRING.sub('', source.decode()).splitlines() if _HEADER_NAMES.search(line)]
    _require(routes == expected, 'producer reads its profile header outside the admitted I/O routes: ' + stage)


def _render_stage(old, stage, doc, before, after):
    previous = (old / stage / 'main.cpp').read_bytes()
    _metadata_only(previous, stage, before)
    a, b = (', '.join(map(str, header)).encode() for header in (before, after))
    _require(previous.count(a) == 1, 'legacy program does not have one isolated wire-header literal: ' + stage)
    rewritten = previous.replace(a, b)
    if stage != 'coo':
        key = 'header' if stage == 'topology' else 'input_header'
        _require(bytes(doc['plan'][key]) == before, 'retained plan disagrees with its wire: ' + stage)
        doc['plan'][key] = list(after)
        _require(native_mesh_basis.render(stage, doc['plan']) == rewritten, 'native renderer changes more than the header: ' + stage)
    return rewritten


def _stage_build(previous, destination, stage, generated, doc, shared_stage):
    destination.mkdir()
    copy2(previous / 'CMakeLists.txt', destination)
    (destination / 'main.cpp').write_bytes(generated)
    if stage == 'coo':
        copytree(previous / 'vendor', destination / 'vendor')
        expected = ()
        target = 'scar_coo_ingress'
        schema = 'scar.native-coo-member'
    else:
        providers = (shared_stage if shared_stage is not None else previous) / 'lib'
        stage_cpu_aten_libraries(destination, providers, pins=doc['libraries'], link=shared_stage is not None)
        expected = tuple(destination / row['uri'] for row in doc['libraries'])
        target = Path(doc['executable']['uri']).name
    executable, linked = build_native_project(destination, target, shlex.split(os.environ.get('CXX', 'c++')),
        expected_libraries=expected, require_cpu_only=stage != 'coo')
    relative = 'bin/' + target
    if stage == 'coo':
        doc['members'][relative] = _local(destination, relative, schema)
        doc['members']['main.cpp'] = _local(destination, 'main.cpp', schema)
        doc['dependencies'] = linked
    else:
        doc['executable'] = _local(destination, relative, 'scar.native-executable-file')
        doc['source'] = _local(destination, 'main.cpp', 'scar.cpp-source')
        doc['linker_members'] = sorted(linked)
    return executable, f'{stage}/{relative}'


def export(publication_path, source_seed_path, *, count, recipe_id, out, solver_stream=None):
    from scar.codegen import native_publication_delivery

    old = Path(publication_path).resolve(strict=True)
    old_doc = native_publication_delivery.check(old)
    old_ref = artifact_reference(old / 'artifact.json', native_publication_delivery.SCHEMA)
    _require(old_doc['schema_version'] == 6 and 'source_seed' in old_doc,
        'upstream rebinding requires an admitted source-sealed publication anchor')
    archive = {**old_doc['source_seed'], 'uri': str(old / old_doc['source_seed']['uri'])}
    original = read_artifact(archive)
    source_ref = artifact_reference(Path(source_seed_path).resolve(strict=True), source_triangle_seed.SCHEMA)
    derived = source_triangle_seed.check(source_ref['uri'])
    _require(all(derived[key] == original[key] for key in _SOURCE_INVARIANTS)
        and derived['iterations'] == count and derived['argv'] == [str(count)],
        'new source profile must independently seal the same count-independent geometry and constants')
    legacy = [json.loads(_actual(old, old_doc[role])) for role in (*_NAMES[:-1], 'ingress')]
    _require(all(row['schema_version'] == 1 for row in legacy), 'count rebinding currently requires the dense admitted pipeline')
    old_wire_refs = [legacy[0]['admitted_input'], *(row[key] for row, key in zip(legacy, _OUTPUT_PORTS, strict=True))]
    _require(all(row[key] == old_wire_refs[index] for index, (row, key) in enumerate(zip(legacy, _INPUT_PORTS, strict=True))),
        'legacy producer and consumer ports disagree')
    old_release = read_artifact(artifact_reference(old / 'binary/release.json', 'scar.sparse-parallel-entry-selection'))
    old_seed = source_triangle_seed_wire.wire_payload(original, legacy[0], old_release)
    old_header = old_seed[:_SIZES[0]]
    target_header = _rebind(old_header, count, recipe_id)
    _, _, n, nnz, _, _ = _CSR.unpack(target_header[-_CSR.size:])
    stream_ref = None
    if solver_stream is not None:
        stream_ref = artifact_reference(Path(solver_stream).resolve(strict=True), SOURCE_STREAM)
        stream = read_artifact(stream_ref)
        header = stream['header']
        _require(stream['recipe_id'] == recipe_id and (header['size'], header['nnz'], header['count'], header['flags'])
            == (n, nnz, count, 0) and header['recipe_digest_hex'] == recipe_id.removeprefix('recipe-v1:')
            and header['format'] == _CSR.format and stream['input']['magic'] == 'SCARPSR1',
            'selected solver ABI differs from the upstream target')

    out = Path(out).resolve()
    _require(not out.exists(), 'native upstream output must be fresh')
    with stage_native_package(out) as root:
        (root / 'wires').mkdir()
        (root / 'source-seed.json').write_bytes(read_artifact(source_ref, json_document=False))
        profile = dict(schema=PROFILE, schema_version=1, recipe_id=recipe_id, case=dict(size=n, nnz=nnz, call_count=count),
            legacy_header=list(old_header), target_header=list(target_header), count_source_verified=True,
            target_solver_stream=stream_ref, scope='NATIVE_UPSTREAM_HEADER_REBIND_ONLY')
        _json(root / 'profile.json', profile)
        profile_ref = _local(root, 'profile.json', PROFILE)
        payload = target_header + old_seed[len(old_header):]
        (root / _WIRES[0]).write_bytes(payload)
        wires = [_local(root, _WIRES[0], old_wire_refs[0]['schema'])]
        comparison = dict(seed=sha256(payload[_SIZES[0]:]).hexdigest())
        commands, packages = {}, {}
        for index, (stage, doc) in enumerate(zip(_NAMES, legacy, strict=True)):
            size, next_size = _SIZES[index:index+2]
            previous = old / stage
            destination = root / stage
            generated = _render_stage(old, stage, doc, old_header[-size:], target_header[-size:])
            native, relative = _stage_build(previous, destination, stage, generated, doc, None if index == 0 else root / 'topology')
            with (root / _WIRES[index]).open('rb') as operand:
                run = subprocess.run([native], stdin=operand, capture_output=True, env=_ENV, timeout=20, check=True)
            prior = read_artifact(old_wire_refs[index+1], json_document=False)
            _require(not run.stderr and prior.startswith(old_header[-next_size:]) and run.stdout.startswith(target_header[-next_size:])
                and run.stdout[next_size:] == prior[next_size:], 'rebuilt native stage changed a numerical body: ' + stage)
            wire_name = _WIRES[index+1]
            (root / wire_name).write_bytes(run.stdout)
            wires.append(_local(root, wire_name, old_wire_refs[index+1]['schema']))
            comparison[stage] = sha256(run.stdout[next_size:]).hexdigest()
            doc[_INPUT_PORTS[index]] = wires[index]
            doc[_OUTPUT_PORTS[index]] = wires[index+1]
            doc['profile_binding'] = profile_ref
            if stage == 'coo':
                doc['stream'] = stream_ref if stream_ref is not None else profile_ref
            _json(destination / 'artifact.json', doc)
            commands[stage] = [relative]
            packages[stage] = _local(root, f'{stage}/artifact.json', doc['schema'])
        result = dict(schema=SCHEMA, schema_version=1, base_publication=old_ref,
            source_seed=_local(root, 'source-seed.json', source_triangle_seed.SCHEMA), profile=profile_ref,
            count=count, recipe_id=recipe_id, case=profile['case'], source_count_verified=True,
            packages=packages, commands=commands, seed_input=wires[0], canonical_csr=wires[-1], wires=wires,
            identical_payload_sha256=comparison, measured_speedup=False, full_python_entry=False)
        _json(root / 'artifact.json', result)
        check(root)
    return artifact_reference(out / 'artifact.json', SCHEMA)


def check(path):
    root = Path(path).resolve(strict=True)
    _, doc = open_native_package(root, SCHEMA)
    profile = json.loads(_actual(root, doc['profile']))
    source = json.loads(_actual(root, doc['source_seed']))
    old, target = bytes(profile['legacy_header']), bytes(profile['target_header'])
    _require(doc['schema_version'] == profile['schema_version'] == 1 and doc['full_python_entry'] is False
        and doc['count'] == profile['case']['call_count'] and doc['recipe_id'] == profile['recipe_id']
        and doc['case'] == profile['case'] and doc['source_count_verified'] is profile['count_source_verified'] is True
        and (source['argv'], source['iterations']) == ([str(doc['count'])], doc['count'])
        and target == _rebind(old, doc['count'], doc['recipe_id'])
        and [row['uri'] for row in doc['wires']] == list(_WIRES)
        and (doc['seed_input'], doc['canonical_csr']) == (doc['wires'][0], doc['wires'][-1]),
        'native upstream profile differs from its exact envelope')
    for wire, size, role in zip(doc['wires'], _SIZES, ('seed', *_NAMES), strict=True):
        data = _actual(root, wire)
        _require(data.startswith(target[-size:]) and sha256(data[size:]).hexdigest() == doc['identical_payload_sha256'][role],
            'local native wire differs from its verified body: ' + role)
    for index, stage in enumerate(_NAMES):
        directory = root / stage
        _require(directory.resolve(strict=True) == directory, 'expected a local native upstream stage: ' + stage)
        if stage == 'coo':
            reference, bound = native_coo_ingress.read_package(directory)
        else:
            reference, bound = native_mesh_basis.read_package(directory, stage)
        _metadata_only((directory / 'main.cpp').read_bytes(), stage, target[-_SIZES[index]:])
        _require(doc['packages'][stage] == {**reference, 'uri': f'{stage}/artifact.json'}
            and bound['profile_binding'] == doc['profile'] and bound[_INPUT_PORTS[index]] == doc['wires'][index]
            and bound[_OUTPUT_PORTS[index]] == doc['wires'][index+1]
            and doc['commands'][stage] == [stage + '/' + (bound['executable']['uri'] if stage != 'coo' else 'bin/scar_coo_ingress')],
            'native upstream stage is not bound to its checked executable and wire ports')
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--publication', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--source-seed', type=Path)
    parser.add_argument('--count', type=int)
    parser.add_argument('--recipe-id')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--solver-stream', type=Path)
    args = parser.parse_args(argv)
    if args.check is not None:
        print(json.dumps(dict(verified=True, count=check(args.check)['count'])))
    elif all(value is not None for value in (args.source_seed, args.count, args.recipe_id, args.out)):
        print(json.dumps(export(args.publication, args.source_seed, count=args.count, recipe_id=args.recipe_id,
            out=args.out, solver_stream=args.solver_stream)))
    else:
        parser.error('export requires --source-seed, --count, --recipe-id and --out')


if __name__ == '__main__':
    main()

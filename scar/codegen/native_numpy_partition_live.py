"""Bridge an exact live ndarray layout to an already-qualified stock CPU schedule."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sysconfig
from tempfile import TemporaryDirectory

from scar.analysis import numpy_partition_live_profiles as profiles
from scar.codegen import native_numpy_complex as stock
from scar.codegen import native_numpy_partition as core
from scar.codegen.native_cpu_elf import cpu_executable_dynamic
from scar.codegen.partition_multiply import _python_target
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.native_numpy_partition import ABI_VERSION, EXTENSION_SCHEMA, MODULE, load

SCHEMA = 'scar.native-numpy-partition-live'
SOURCE = 'src/live_partition.cpp'
_TEMPLATE = Path(__file__).with_suffix('.cpp.in')
_ROLES = ('target', 'matrix', 'out')
_FLAGS = ('-std=c++20', '-O3', '-fPIC', '-shared', '-fvisibility=hidden', '-fno-fast-math', '-ffp-contract=off')
_KEYS = [{'kind': 'ELLIPSIS_LAST_AXIS_INDEX', 'index': i, 'normalized_index': i} for i in range(2)]
_REPACK = 'REPACK_MISMATCHED_RESIDUES_ONLY'
_BASE_WORKSPACE = 'temp_offset + rounded(region.sizes[3] + region.residues[3])'


def _need(ok, reason):
    if not ok:
        raise ValueError('native live NumPy partition: ' + reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _workspace(profile, names):
    sizes, residues = profile['sizes'], profile['residues']
    return sum((sizes[name] + residues[name] + 63) & ~63 for name in names)


def _layout_constants(profile):
    declarations, records = [], []
    for role in _ROLES:
        geometry = profile['geometry'][role]
        shape, strides = geometry['shape'], geometry['strides_bytes']
        _need(geometry['dtype'] == '<c16' and 0 < len(shape) == len(strides) <= 64 and
              all(type(value) is int and 0 < value <= core._MAX for value in shape) and
              all(type(value) is int and 0 <= value <= core._MAX for value in strides),
              'source role must supply a bounded positive native complex geometry')
        extent = 16
        for stride, dimension in sorted(zip(strides, shape, strict=True)):
            if dimension == 1:
                continue
            _need(stride == extent, 'source role has storage holes or overlapping elements')
            extent *= dimension
        native_name = 'output' if role == 'out' else role
        _need(extent == profile['sizes'][native_name], 'physical root size differs from the dense array view')
        for key, values in (('shape', shape), ('strides', strides)):
            literal = ', '.join(map(str, values))
            declarations.append(f'constexpr std::array<std::int64_t, {len(values)}> {role}_{key}{{{literal}}};')
        records.append(f'{{{role}_shape, {role}_strides}}')
    declarations.append('constexpr Layout geometry[]{' + ', '.join(records) + '};')
    return '\n'.join(declarations)


def _part(source, begin, end):
    _need(source.count(begin) == source.count(end) == 1 and source.index(begin) < source.index(end),
          'shared native region implementation changed')
    return begin + source.split(begin, 1)[1].split(end, 1)[0].rstrip()


def _render(profile, provider, repack):
    _need(profile['implementation'] == 'elided-copy' and profile['selectors'] == _KEYS and
          profile['rounding_mode_raw'] == 0 and profile['geometry']['target'] == profile['geometry']['out'],
          'source scope does not match the qualified private publication route')
    steps = profile['steps']
    _need(steps and all(row['op'] in ('ADD', 'MULTIPLY') and
              [arg['stride'] for arg in row['operands']] in provider['strides'][row['op'].lower()] for row in steps),
          'source schedule differs from the live stock loop selection')
    body, region = core._region_lines(0, profile['id'], profile['sizes'], profile['residues'], steps)
    source = core._TEMPLATE.read_text()
    inputs = dict(CORE_TYPES=_part(source, 'enum class Op', '@PLANS@'),
                  PLANS=body + '\nconstexpr Region regions[]{' + region + '};',
                  GUARDS=_layout_constants(profile),
                  EXECUTE=_part(source, 'void execute(', '\nconst Region& get('), MODULE=MODULE,
                  INPUT_PLACEMENT='        (role == 2 || storage.first % 64 == region.residues[role]) &&\n',
                  WORKSPACE=f'    constexpr auto workspace = {_BASE_WORKSPACE};\n',
                  ROOTS_QUALIFIER='const ', REPACK='')
    if repack:
        inputs.update(INPUT_PLACEMENT='', ROOTS_QUALIFIER='',
                      WORKSPACE=f'    auto workspace = {_BASE_WORKSPACE};\n' + '''    const std::array<Interval, 2> borrowed{target, matrix};
    std::array<std::size_t, borrowed.size()> repack_offsets{};
    for (std::size_t role = 0; role < borrowed.size(); ++role) {
        if (borrowed[role].first % 64 != region.residues[role]) {
            repack_offsets[role] = workspace;
            workspace += rounded(region.sizes[role] + region.residues[role]);
        }
    }
''', REPACK='''    for (std::size_t role = 0; role < borrowed.size(); ++role) {
        if (repack_offsets[role]) {
            auto* destination = bytes + repack_offsets[role] + region.residues[role];
            std::memcpy(destination, roots[role], region.sizes[role]);
            roots[role] = destination;
        }
    }
''')
    return re.sub(r'@([A-Z_]+)@', lambda match: inputs[match[1]], _TEMPLATE.read_text()).encode()


def _environment(authority, provider):
    import numpy as np
    from numpy._core import _multiarray_umath as numpy_extension

    python = _python_target()
    numpy = authority['numpy']
    original_python = authority['source_python']
    language = {**original_python, 'uri': str(Path(original_python['uri']).resolve(strict=True))}
    image = _ref(numpy_extension.__file__, numpy['extension']['schema'])
    _need({**python['executable'], 'schema': original_python['schema']} == language and
          image == numpy['extension'] == {**provider['provider'], 'schema': image['schema']} and
          _ref(np.__file__, numpy['module']['schema']) == numpy['module'] and
          np.__version__ == numpy['version'] == provider['numpy_version'] and
          numpy_extension._get_ndarray_c_version() == provider['ndarray_c_abi'],
          'compiler interpreter or stock NumPy C API differs from the witnessed image')
    pyinclude, npinclude = (Path(path).resolve(strict=True) for path in (sysconfig.get_path('include'), np.get_include()))
    headers = [pyinclude / name for name in ('Python.h', 'pyconfig.h')] + sorted((npinclude / 'numpy').rglob('*.h'))
    return dict(python=python, numpy=numpy, ndarray_c_abi=provider['ndarray_c_abi'],
                includes=[str(pyinclude), str(npinclude)],
                headers=[_ref(path, 'scar.cpp-header') for path in headers])


def _prepare(profile_ref, provider_ref, repack):
    authority = profiles.check(profile_ref['uri'])
    _need(authority == read_artifact(profile_ref) and authority['provider'] == provider_ref and
          authority['deployment_authorized'] is False, 'profile or provider identity changed')
    _need(not repack or authority['guards']['input_placement'] ==
          'MATCHED_INPUT_ADDRESS_RESIDUES_OR_PRIVATE_REPACK_AT_MATCHED_RESIDUES',
          'source contract does not admit alternative private input placement')
    provider = read_artifact(provider_ref)
    files = stock.render(provider)
    files[SOURCE] = _render(authority['profile'], provider, repack)
    return authority, provider, files, _environment(authority, provider)


def _linkage(extension, provider):
    dynamic = cpu_executable_dynamic(extension)
    _need(dynamic['runpath'] is None and dynamic['rpath'] is None and
          set(dynamic['needed']) <= {'libc.so.6', 'libstdc++.so.6', 'libm.so.6', 'libgcc_s.so.1'},
          'live extension has an unqualified dynamic provider dependency')
    return dict(dynamic=dynamic, stock=stock.audit(extension, provider))


def _array(geometry, data, residue):
    import numpy as np

    owner = np.empty(len(data) + 63, dtype=np.uint8)
    first = (residue - owner.ctypes.data) % 64
    owner[first:first+len(data)] = np.frombuffer(data, dtype=np.uint8)
    return np.ndarray(tuple(geometry['shape']), dtype=np.dtype(geometry['dtype']), buffer=owner, offset=first,
                      strides=tuple(geometry['strides_bytes']))


def _validate(extension, profile, repack):
    import numpy as np

    module = load(extension)
    data = {name: read_artifact(reference, json_document=False) for name, reference in profile['inputs'].items()}
    expected = read_artifact(profile['expected'], json_document=False)
    offsets = [(0, 0), (16, 0), (0, 16), (16, 16)] if repack else [(0, 0)]
    placements = []
    for shifts in offsets:
        inputs = {name: (profile['residues'][name] + shift) % 64
                  for name, shift in zip(('target', 'matrix'), shifts, strict=True)}
        target, matrix = (_array(profile['geometry'][name], data[name], residue) for name, residue in inputs.items())
        out = _array(profile['geometry']['out'], b'\xa5' * len(expected), 0)
        with np.errstate(**profile['numpy_errstate']):
            result = module.try_apply(target, matrix, ((Ellipsis, 0), (Ellipsis, 1)), out)
        actual = out.tobytes(order='A') if out.flags.c_contiguous else (
            np.ndarray((len(expected),), dtype=np.uint8, buffer=out.base,
                       offset=out.ctypes.data - out.base.ctypes.data).tobytes())
        _need(result is out and actual == expected, 'live publication does not match original witnessed result and identity')
        placements.append(inputs)
    validation = dict(sample=profile['source_sample'], original=profile['expected'], bytes=len(expected),
                      sha256=sha256(expected).hexdigest(), exact_original_storage=True, returned_exact_out=True)
    if repack:
        validation['verified_input_address_residues'] = placements
    return validation


def _manifest(profile_ref, provider_ref, authority, environment, files, extension, linkage, validation, repack):
    profile = authority['profile']
    base_bytes = _workspace(profile, ('output', 'temp'))
    result = dict(schema=SCHEMA, schema_version=1, profile=profile_ref, provider=provider_ref, extension=extension,
                module=MODULE, abi_version=ABI_VERSION, environment=environment, files=files, linkage=linkage,
                validation=validation, cpu_ids=authority['cpu_ids'], cxx_standard=20,
                private_workspace_bytes=base_bytes, whole_program_equivalence=False,
                runtime='SCOPED_CPYTHON_NDARRAY_LIVE_BRIDGE', output_publication='PRIVATE_RESULT_FULL_COPY',
                floating_environment='RESTORE_CALLER_ENVIRONMENT_BEFORE_RETURN',
                requires_unobserved_source_sticky_status=True,
                dynamic_eligibility='GUARDED_TRY_APPLY', caller_provider_and_errstate_guard_required=True,
                deployment_authorized=False)
    if repack:
        extra_bytes = _workspace(profile, ('target', 'matrix'))
        result.update(input_materialization=_REPACK, private_workspace_bytes=base_bytes + extra_bytes,
                      matched_input_workspace_bytes=base_bytes, max_extra_input_workspace_bytes=extra_bytes,
                      max_input_copy_bytes=sum(profile['sizes'][name] for name in ('target', 'matrix')),
                      input_placement_guard=dict(residue_comparisons=2, max_private_segments=2,
                                                 additional_allocation_calls=0, isolated_timing_measured=False))
    return result


def export(profile_path, provider_path, out, *, repack=False):
    profile_ref, provider_ref = _ref(profile_path, profiles.SCHEMA), _ref(provider_path, stock.SCHEMA)
    authority, provider, files, environment = _prepare(profile_ref, provider_ref, repack)
    out = Path(out).absolute()
    _need(not out.exists() and not out.is_symlink(), 'fresh extension package path required')
    out.parent.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    _need(compiler, 'C++20 compiler unavailable')
    with TemporaryDirectory(prefix='.scar-live-numpy-partition-', dir=out.parent) as temp:
        root = Path(temp) / 'package'
        for name, payload in files.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        basename = MODULE + environment['python']['extension_suffix']
        command = [*compiler, *_FLAGS, '-I', 'include/scar_generated']
        for path in environment['includes']:
            command.extend(('-I', path))
        command.extend((SOURCE, stock.SOURCE, '-o', basename))
        env = {key: value for key, value in os.environ.items() if not key.startswith('LD_')}
        subprocess.run(command, cwd=root, env=env, check=True, timeout=60)
        extension = _ref(root / basename, EXTENSION_SCHEMA)
        linkage = _linkage(extension['uri'], provider)
        validation = _validate(extension, authority['profile'], repack)
        members = {key: {**_ref(root / key, 'scar.native-live-partition-member'), 'uri': str(out / key)}
                   for key in files}
        result = _manifest(profile_ref, provider_ref, authority, environment, members,
                           {**extension, 'uri': str(out / basename)}, linkage, validation, repack)
        (root / 'result.json').write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
        root.rename(out)
    return result


def check(path):
    path = Path(path).resolve(strict=True)
    stored = read_artifact(_ref(path, SCHEMA))
    profile, provider = (stored[key] for key in ('profile', 'provider'))
    repack = stored.get('input_materialization') == _REPACK
    authority, original, files, environment = _prepare(profile, provider, repack)
    root = path.parent
    extension = _ref(root / (MODULE + environment['python']['extension_suffix']), EXTENSION_SCHEMA)
    linkage = _linkage(extension['uri'], original)
    members = {key: _ref(root / key, 'scar.native-live-partition-member') for key in files}
    _need(all(read_artifact(members[key], json_document=False) == payload for key, payload in files.items()),
          'generated extension source or stock member differs')
    validation = _validate(extension, authority['profile'], repack)
    _need(stored == _manifest(profile, provider, authority, environment, members, extension, linkage, validation, repack),
          'live extension receipt differs during cold replay')
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ('profile', 'provider', 'out', 'check'):
        parser.add_argument('--' + role, type=Path)
    parser.add_argument('--repack', action='store_true')
    args = parser.parse_args(argv)
    if args.check and not any((args.profile, args.provider, args.out, args.repack)):
        result = check(args.check)
    elif not args.check and all((args.profile, args.provider, args.out)):
        result = export(args.profile, args.provider, args.out, repack=args.repack)
    else:
        parser.error('provide --check RECEIPT or --profile PROFILE --provider STOCK --out PACKAGE')
    print(json.dumps(dict(module=result['module'], extension=result['extension'], validation=result['validation'])))


if __name__ == '__main__':
    main()

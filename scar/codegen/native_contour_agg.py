"""Lower exact AGG path slots to runtime ContourPy geometry in C++20."""

import argparse
import json
from pathlib import Path
import shutil
import struct
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen import native_contour_scene
from scar.codegen.native_project import build_native_project, write_native_project
from scar.compiler.plan_contour_scene import SCHEMA as PLAN, read_contour_scene_plan
from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.native-contour-agg'
_TARGET = 'scar_contour_agg'
_EXE = 'bin/' + _TARGET
_SOURCES = ('src/main.cpp', 'src/contour_scene.cpp', 'vendor/contour_generator.cpp', 'vendor/mpl2014.cpp')


def _require(ok, detail):
    if not ok:
        raise ValueError('native contour AGG: ' + detail)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _root_bindings(lineage, shape):
    if lineage is None:
        return None
    _require(lineage['sampled_alias_relation'] is True and set(lineage['bindings']) == {'x', 'y', 'z'},
             'exact observed original contour roots required')
    read_artifact(lineage['source'], json_document=False)
    for route in lineage['bindings'].values():
        rows, cols = route['root_shape']
        row, col = route['start']
        field = route['field_path']
        _require(route['shape'] == shape and route['byte_strides'] == [8 * cols, 8]
                 and 0 <= row <= rows - shape[0] and 0 <= col <= cols - shape[1]
                 and len(field) >= 2 and all(name.isidentifier() for name in field),
                 'geometry differs from the observed original rectangular view')
    return lineage['bindings']


def _template(sample, observation):
    original = read_artifact(sample['scene'], json_document=False)
    grid = sample['grid']
    lineage = observation['grid'].get('lineage')
    roots = _root_bindings(lineage, grid['shape'])
    rows, cols = grid['shape']
    cy, cx = sample['provider']['chunk_size']
    _require(sample['provider']['masked'] is False, 'scene requires an original unmasked input')
    levels = [sample['intervals'][0]['lower'], *(row['upper'] for row in sample['intervals'])]
    output = bytearray(b'SCARCTT1' + struct.pack('<6Q', rows, cols, cx, cy,
                        sample['provider']['corner_mask'], len(levels)))
    output.extend(struct.pack(f'<{len(levels)}d', *levels))
    output.extend(struct.pack('<Q', 3 if roots else 0))
    if roots:
        for axis in ('x', 'y', 'z'):
            route = roots[axis]
            output.extend(struct.pack('<4Q', *route['root_shape'], *route['start']))
    replacements = sorted(sample['collections'], key=lambda row: row['path_span']['start'])
    output.extend(struct.pack('<Q', len(replacements)))
    cursor = 0
    removed = 0
    for row in replacements:
        start, end = (row['path_span'][key] for key in ('start', 'end'))
        _require(cursor <= start < end <= len(original) and not any(row['path_simplifies']),
                 'admitted closed-polygon scene spans overlap or require a simplifier')
        output.extend(struct.pack('<Q', start - cursor) + original[cursor:start])
        output.extend(struct.pack('<I', len(row['bands'])))
        for band, threshold in zip(row['bands'], row['simplify_threshold'], strict=True):
            output.extend(struct.pack('<Qd', band, threshold))
        removed += end - start
        cursor = end
    output.extend(struct.pack('<Q', len(original) - cursor) + original[cursor:])
    return bytes(output), removed, roots


def _prepare(plan_ref, provider_ref):
    plan = read_contour_scene_plan(plan_ref['uri'])
    _require(plan == read_artifact(plan_ref) and plan['status'] == 'CONDITIONAL_EXPLICIT_TRIAL_ONLY'
             and plan['whole_entry_authorized'] is False, 'requires the exact scoped contour scene plan')
    provider_path = Path(provider_ref['uri']).parent
    provider = native_contour_scene.check(provider_path)
    _require(provider == read_artifact(provider_ref) and provider['algorithm'] == 'mpl2014',
             'original native contour provider receipt differs')
    members = {}
    for name, reference in provider['files'].items():
        if name not in ('CMakeLists.txt', native_contour_scene._EXECUTABLE, 'src/main.cpp'):
            _require(reference['uri'] == name, 'native contour member path differs')
            members[name] = read_artifact({**reference, 'uri': str(provider_path / name)}, json_document=False)
    members['src/main.cpp'] = Path(__file__).with_name('native_contour_agg_v1.cpp.in').read_bytes()
    observation = read_artifact(plan['observation'])
    samples = []
    for index, sample in enumerate(plan['samples']):
        provider_version = sample['provider']['version']
        _require(provider_version == provider['contourpy_version'], 'native contour provider version differs')
        suffix = sample['source_observation']['locator'].removeprefix('#/samples/')
        raw, removed, roots = _template(sample, observation['samples'][int(suffix)])
        output = f'scenes/sample-{index}.ctt'
        members[output] = raw
        samples.append(dict(frame_index=sample['frame_index'], template=output,
                            exact_source_scene=sample['scene'], input=sample['input'],
                            replaced_path_bytes=removed, template_bytes=len(raw), original_root_views=roots,
                            contains_recorded_target_geometry=False,
                            styles_and_levels_from_original_observation=True))
    return samples, members


def _project(root):
    write_native_project(root, _TARGET, sources=_SOURCES, include_dirs=('vendor',),
        compile_options=('-fno-fast-math', '-ffp-contract=off'),
        install_dirs=(('licenses', '.'), ('scenes', '.')))


def _result(root, plan, provider, samples, members, libraries):
    names = (*members, _EXE, 'CMakeLists.txt')
    return dict(schema=SCHEMA, schema_version=1, plan=plan, provider=provider, samples=samples,
                files={name: {**_ref(root / name, 'scar.native-contour-agg-member'), 'uri': name} for name in names},
                same_original_draw_tapes=True, linked_libraries=sorted(libraries),
                cxx_standard=20, python_runtime_required=False, fixed_residual_scene_inputs=True,
                whole_program_translated=False, deployment_authorized=False)


def export(plan, provider, out):
    plan_ref = _ref(plan, PLAN)
    provider_path = Path(provider).resolve(strict=True)
    provider_ref = _ref(provider_path / 'result.json', native_contour_scene.SCHEMA)
    samples, members = _prepare(plan_ref, provider_ref)
    out = Path(out).absolute()
    _require(not out.exists() and not out.is_symlink(), 'fresh contour AGG package directory required')
    compiler = shutil.which('g++')
    _require(compiler is not None, 'GNU C++20 compiler unavailable')
    with TemporaryDirectory(prefix='.scar-contour-agg-', dir=out.parent) as temp:
        root = Path(temp) / 'package'
        for name, data in members.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        _project(root)
        exe, libraries = build_native_project(root, _TARGET, (compiler,))
        for index, sample in enumerate(samples):
            scene = Path(temp) / f'regenerated-{index}.agg'
            subprocess.run([str(exe), str(root / sample['template']), str(scene), sample['input']['uri']],
                           env={}, check=True, timeout=20)
            _require(scene.read_bytes() == read_artifact(sample['exact_source_scene'], json_document=False),
                     'native geometry did not regenerate the original drawing tape')
        report = _result(root, plan_ref, provider_ref, samples, members, libraries)
        (root / 'result.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
        root.rename(out)
    return report


def check(package):
    root = Path(package).resolve(strict=True)
    report = read_artifact(_ref(root / 'result.json', SCHEMA))
    samples, members = _prepare(report['plan'], report['provider'])
    _require(report == _result(root, report['plan'], report['provider'], samples, members, report['linked_libraries']),
             'native contour AGG package differs during replay')
    for name, data in members.items():
        _require((root / name).read_bytes() == data, 'native template member bytes differ: ' + name)
    with TemporaryDirectory(prefix='scar-contour-agg-layout-') as temp:
        _project(Path(temp))
        _require((Path(temp) / 'CMakeLists.txt').read_bytes() == (root / 'CMakeLists.txt').read_bytes(),
                 'native contour AGG project differs')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('check', 'plan', 'provider', 'out'):
        parser.add_argument('--' + name)
    args = parser.parse_args(argv)
    if args.check:
        report = check(args.check)
    elif args.plan and args.provider and args.out:
        report = export(args.plan, args.provider, args.out)
    else:
        parser.error('provide --check PACKAGE or --plan PLAN --provider PACKAGE --out OUTPUT')
    print(json.dumps(dict(samples=len(report['samples']), exact_draw_tapes=True)))


if __name__ == '__main__':
    main()

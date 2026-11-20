"""Calibrate complete synchronous rendering against the two released native plot ABIs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen import native_live_plot as plot
from scar.codegen import native_live_plot_resident as resident
from scar.codegen.native_png_service import _digest_text
from scar.codegen.native_project import build_native_project, write_native_project
from scar.ir.execution_recipe import read_artifact

SCHEMA = 'scar.native-live-plot-service'
RESOURCE_SCHEMA = 'scar.native-live-plot-service-resources'
RESOURCE_ENV = 'SCAR_PLOT_SERVICE_RESOURCE_LOG'
TARGET = 'scar_live_plot_service'
EXECUTABLE = 'bin/' + TARGET
ROLE_PATHS = {'executable': EXECUTABLE, 'layout': plot.LAYOUT, 'font': plot.FONT}
_SOURCE = 'src/main.cpp'
_CODE = Path(__file__).parent
_VARIANTS = ('v1', 'v2')


def _require(valid, detail):
    if not valid:
        raise ValueError('native live plot service: ' + detail)


def _source(files, host_paths, demo, variant):
    preamble = plot._prefix(demo.decode(), 'int main(' if variant == 'v1' else 'struct Prepared {')
    loader = plot._prefix((_CODE / 'native_png_service.cpp.in').read_text(), 'struct RuntimeBinding')
    start = 'Digest hash(std::span<const unsigned char> data)'
    _require(loader.count(start) == 1, 'original native file and loader helpers are missing')
    loader = plot._replace(loader[loader.index(start):], '    link_map* image() const {',
        '    void* get() const noexcept { return handle_; }\n    link_map* image() const {')
    quoted = lambda value: json.dumps(str(value), ensure_ascii=True)
    members = ['{' + quoted(name) + f', {len(data)}, {{{_digest_text(sha256(data).hexdigest())}}}' + '}'
               for name, data in files.items() if name.startswith('lib/')]
    fields = dict(ROOT_SUPPORT=preamble, BOUND_LOADER=loader, RESIDENT=int(variant == 'v2'),
                  PROVIDER=quoted(plot.LIBRARY), FILE_COUNT=len(members), FILES=',\n    '.join(members),
                  HOST_COUNT=len(host_paths), HOSTS=',\n    '.join(map(quoted, host_paths)))
    template = (_CODE / 'native_live_plot_service.cpp.in').read_text()
    return re.sub(r'@([A-Z_]+)@', lambda match: str(fields[match[1]]), template).encode()


def _prepare(root, release, variant):
    reference = plot._ref(root / 'result.json', release['schema'])
    expected = plot.SCHEMA if variant == 'v1' else resident.SCHEMA
    _require(release == read_artifact(reference) and release['schema'] == expected and
             release['exact_original_png_bytes'] is True and release['python_runtime_required'] is False,
             'expected the exact released native plot arm')
    private = set(release['linkage']['private'])
    libraries = {plot.LIBRARY, *(f'lib/{name}' for name in private)}
    _require({'lib/libz.so.1', plot.LIBRARY} <= libraries, 'released PNG closure differs')
    resolved = plot._ldd(root / plot.DEMO)
    host_paths = sorted(resolved[name] for name in release['linkage']['system'] if name != 'linux-vdso.so.1')
    _require(all(Path(path).is_absolute() for path in host_paths) and len(host_paths) == len(set(host_paths)) and
             any(Path(path).name == 'libcrypto.so.3' for path in host_paths), 'released host closure differs')
    abi = 'include/scar_generated/native_live_plot_abi_v1.h' if variant == 'v1' else resident.ABI
    members = libraries | {abi, 'include/scar_generated/scar_abi_v1.h',
                           'include/scar_generated/live_plot_profile.hpp', plot.LAYOUT, plot.FONT}
    members.update(name for name in release['files'] if name.startswith('licenses/'))
    files = {name: plot._member(root, release, name) for name in sorted(members)}
    files[_SOURCE] = _source(files, host_paths, plot._member(root, release, 'src/demo.cpp'), variant)
    return reference, files, host_paths


def _project(root):
    write_native_project(root, TARGET, libraries=('crypto', 'dl'),
                         compile_options=('-fno-fast-math', '-ffp-contract=off'),
                         install_dirs=(('assets', '.'), ('lib', '.'), ('licenses', '.')))


def _rows(output, variant, frames):
    lines = output.decode('ascii').splitlines()
    _require(lines and lines[0] == 'operation\tframe\telapsed_ns', 'clean timing TSV header differs')
    expected = [('render', idx) for idx in range(frames)]
    if variant == 'v2':
        expected = [('prepare', -1), *expected, ('destroy', -1)]
    triplets = [row.split('\t') for row in lines[1:]]
    _require(len(triplets) == len(expected), 'clean timings do not include each released call once')
    for row, (operation, frame) in zip(triplets, expected, strict=True):
        _require(len(row) == 3 and row[:2] == [operation, str(frame)] and re.fullmatch('[1-9][0-9]*', row[2]),
                 'clean timing row or released call order differs')


def _validate(root, cases, variant):
    _require(len(cases) >= 3 and all(len(row['roots']) == 3 and row['roots'][:2] == cases[0]['roots'][:2]
                                    for row in cases), 'require three identical-coordinate native oracle states')

    def arguments(root):
        read_artifact(root['input'], json_document=False)
        return root['input']['uri'], str(root['offset'])

    with TemporaryDirectory(prefix='scar-render-service-oracle-') as temporary:
        out = Path(temporary)
        resources = out / 'resources.json'
        args = [root / EXECUTABLE, root / plot.LAYOUT, root / plot.FONT,
                *arguments(cases[0]['roots'][0]), *arguments(cases[0]['roots'][1])]
        outputs = []
        for i, row in enumerate(cases):
            target = out / f'frame-{i}.png'
            args.extend((*arguments(row['roots'][2]), row['time'], target))
            outputs.append(target)
        run = subprocess.run(args, env={'LC_ALL': 'C', RESOURCE_ENV: str(resources)}, check=True,
                             capture_output=True, timeout=35)
        _require(not run.stderr, 'clean native driver unexpectedly emitted stderr')
        _rows(run.stdout, variant, len(cases))
        observed = json.loads(resources.read_bytes())
        _require(observed['schema'] == RESOURCE_SCHEMA and observed['schema_version'] == 1 and
                 observed['vm_peak_kib'] >= observed['vm_hwm_kib'] > 0 and observed['ru_maxrss_kib'] > 0,
                 'native process memory witness differs')
        for target, row in zip(outputs, cases, strict=True):
            _require(target.read_bytes() == read_artifact(row['expected'], json_document=False),
                     'clean native driver differs from original PNG')


def _arm_report(root, variant, reference, files, host_paths, validation):
    names = (*files, EXECUTABLE, 'CMakeLists.txt')
    members = {name: {**plot._ref(root / name, 'scar.native-live-plot-service-member'), 'uri': f'{variant}/{name}'}
               for name in names}
    return dict(source_provider=reference, files=members, host_paths=host_paths, validation=validation,
                **{role: members[path] for role, path in ROLE_PATHS.items()})


def _arm(source, release, root, variant):
    reference, files, host = _prepare(source, release, variant)
    for name, payload in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    _project(root)
    compiler = shutil.which('g++')
    _require(compiler is not None, 'GNU C++20 compiler unavailable')
    build_native_project(root, TARGET, (compiler,))
    witness = release['validation'][:3]
    _validate(root, witness, variant)
    return _arm_report(root, variant, reference, files, host, witness)


def _product(arms):
    return dict(schema=SCHEMA, schema_version=1, arms=arms,
                clock='CXX_STEADY_CLOCK_NS_DIRECT_PUBLIC_ABI_CALLS', cxx_standard=20,
                argv=['LAYOUT', 'FONT', 'X_FILE', 'X_OFFSET', 'Y_FILE', 'Y_OFFSET',
                      '(Z_FILE Z_OFFSET TIME FRESH_OUTPUT_PNG)+'],
                stdout_columns=['operation', 'frame', 'elapsed_ns'], resource_env=RESOURCE_ENV,
                resource_schema=RESOURCE_SCHEMA, process_scope_resource_metrics_separate=True,
                stock_libraries_only=True, all_original_pngs_exact=True, python_runtime_required=False,
                packed_inputs_and_path_resolution_excluded=True, deployment_authorized=False)


def _matched(first, v1, v2):
    _require(v2['source_provider'] == first and v1['validation'] == v2['validation'],
             'released plot arms use different original states')


def build(v1, v2, out):
    sources = [Path(value).resolve(strict=True) for value in (v1, v2)]
    releases = [plot.check(sources[0]), resident.check(sources[1])]
    _matched(plot._ref(sources[0] / 'result.json', plot.SCHEMA), *releases)
    package = Path(out).absolute()
    _require(not package.exists() and not package.is_symlink(), 'require a fresh calibration package')
    with TemporaryDirectory(prefix='.scar-render-service-', dir=package.parent) as temporary:
        root = Path(temporary) / 'package'
        root.mkdir()
        with ThreadPoolExecutor(max_workers=2) as pool:
            builds = {name: pool.submit(_arm, source, release, root / name, name)
                      for name, source, release in zip(_VARIANTS, sources, releases, strict=True)}
            arms = {name: pending.result() for name, pending in builds.items()}
        report = _product(arms)
        (root / 'result.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
        root.rename(package)
    return report


def members(root, package):
    """Return absolute exact references for the native runner and its two inputs."""
    root = Path(root).resolve(strict=True)
    _require(package['schema'] == SCHEMA and set(package['arms']) == set(_VARIANTS),
             'unsupported clean native calibration package')
    selected = {}
    for name, arm in package['arms'].items():
        selected[name] = {}
        for role, path in ROLE_PATHS.items():
            reference = arm[role]
            _require(reference == arm['files'][path] and reference['uri'] == f'{name}/{path}',
                     'calibration member role differs')
            exact = {**reference, 'uri': str(root / reference['uri'])}
            read_artifact(exact, json_document=False)
            selected[name][role] = exact
    return selected


def check(package):
    root = Path(package).resolve(strict=True)
    report = read_artifact(plot._ref(root / 'result.json', SCHEMA))
    _require(set(report['arms']) == set(_VARIANTS), 'unsupported clean arms')
    arms, releases = {}, {}
    for name, supplied in report['arms'].items():
        source = Path(supplied['source_provider']['uri']).parent
        release = (plot.check if name == 'v1' else resident.check)(source)
        releases[name] = release
        reference, files, host = _prepare(source, release, name)
        arm = root / name
        actual = _arm_report(arm, name, reference, files, host, release['validation'][:3])
        _require(supplied == actual, 'calibration receipt differs: ' + name)
        for member, data in files.items():
            _require(read_artifact({**actual['files'][member], 'uri': str(arm / member)}, json_document=False) == data,
                     'generated calibration source differs: ' + member)
        with TemporaryDirectory(prefix='scar-render-service-project-') as temporary:
            draft = Path(temporary)
            _project(draft)
            _require((arm / 'CMakeLists.txt').read_bytes() == (draft / 'CMakeLists.txt').read_bytes(),
                     'calibration CMake recipe differs: ' + name)
        arms[name] = actual
    _matched(arms['v1']['source_provider'], releases['v1'], releases['v2'])
    _require(report == _product(arms), 'clean native calibration receipt differs')
    members(root, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('v1', 'v2', 'out', 'check'):
        parser.add_argument('--' + name)
    args = parser.parse_args(argv)
    if args.check and not any((args.v1, args.v2, args.out)):
        report = check(args.check)
    elif not args.check and all((args.v1, args.v2, args.out)):
        report = build(args.v1, args.v2, args.out)
    else:
        parser.error('provide --check PACKAGE or --v1 STOCK --v2 RESIDENT --out FRESH_PACKAGE')
    print(json.dumps(dict(schema=SCHEMA, arms=list(report['arms']), all_original_pngs_exact=True)))


if __name__ == '__main__':
    main()

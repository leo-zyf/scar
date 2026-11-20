"""Reuse one qualified native plot owner across synchronous frame publication."""

import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen import native_live_plot as base
from scar.ir.execution_recipe import read_artifact

SCHEMA = 'scar.native-live-plot-resident'
LIBRARY = base.LIBRARY
LAYOUT = base.LAYOUT
FONT = base.FONT
ABI = 'include/scar_generated/native_live_plot_resident_abi_v2.h'
SYMBOLS = dict(prepare='scar_live_plot_prepare_v2', render='scar_live_plot_render_v2',
               destroy='scar_live_plot_destroy_v2')
_TEMPLATES = Path(__file__).parent


def _require(condition, detail):
    if not condition:
        raise ValueError('native live plot resident: ' + detail)


def _template(name):
    return (_TEMPLATES / ('native_live_plot_resident' + name)).read_text()


def _compose(original):
    source = original.decode()
    preamble = base._prefix(source, 'namespace {\nstd::vector<char> profile_bytes')
    preamble = base._replace(preamble, '#include "native_live_plot_abi_v1.h"\n', '')
    begin = '    const auto [zmin, zmax] = finite_bounds(grid.z);'
    end = '    return std::move(scene).finish();'
    _require(source.count(begin) == source.count(end) == 1 and source.index(begin) < source.index(end),
             'source-derived scene composition boundary changed')
    body = source[source.index(begin):source.index(end) + len(end)]
    body = base._replace(body, '    const auto style = scar::contour::make_style',
                         '    auto style = scar::contour::make_style')
    body = base._replace(body, '    grid.levels = style.contour_levels;',
                         '    grid.levels = std::move(style.contour_levels);')
    return (preamble + base._replace(_template('.cpp.in'), '@COMPOSE_BODY@', body)).encode()


def _font(original):
    source = original.decode()
    proto = 'std::vector<Glyph> generate_glyphs(std::span<const char> bytes, double time)'
    source = base._replace(source, proto, '''struct ResidentFont {
    std::vector<char> bytes;
    Font font;
    explicit ResidentFont(std::vector<char> data) : bytes(std::move(data)), font(bytes) {}
};

std::vector<Glyph> generate_glyphs(ResidentFont& owner, double time)''')
    begin = '    Font provider{bytes};'
    end = '    const auto height = canvas_height;'
    _require(source.count(begin) == source.count(end) == 1 and source.index(begin) < source.index(end),
             'source glyph provider setup boundary changed')
    version = source[source.index(begin) + len(begin) + 1:source.index(end)]
    source = base._between(source, begin, end, '    auto& provider = owner.font;\n')
    marker = 'std::vector<Glyph> generate_glyphs(ResidentFont& owner, double time)'
    source = base._replace(source, marker, '''ResidentFont* prepare_font(std::vector<char> bytes) {
    auto owner = std::make_unique<ResidentFont>(std::move(bytes));
    auto& provider = owner->font;
''' + version + '''    return owner.release();
}

void destroy_font(ResidentFont* owner) { delete owner; }

''' + marker)
    return source.encode()


def _prepare(reference):
    root = Path(reference['uri']).parent
    report = base.check(root)
    _require(report == read_artifact(reference) and report['in_process'] is True and
             report['exact_original_png_bytes'] is True, 'original provider receipt differs')
    files = {name: base._member(root, report, name) for name in report['files']
             if name not in (LIBRARY, base.DEMO, 'CMakeLists.txt') and
             not name.endswith('/native_live_plot_abi_v1.h')}
    files['src/live.cpp'] = _compose(files['src/live.cpp'])
    files['src/glyph.cpp'] = _font(files['src/glyph.cpp'])
    files['src/contour_agg_io.hpp'] = base._replace(
        files['src/contour_agg_io.hpp'].decode(),
        '    bool end() { return total_ == input_.size(); }',
        '    bool end() { return total_ == input_.size(); }\n    std::size_t offset() const { return total_; }').encode()
    files['include/scar_generated/live_plot_internal.hpp'] = base._replace(
        files['include/scar_generated/live_plot_internal.hpp'].decode(),
        'std::vector<Glyph> generate_glyphs(std::span<const char> font, double time);',
        '''struct ResidentFont;
ResidentFont* prepare_font(std::vector<char> bytes);
void destroy_font(ResidentFont* owner);
std::vector<Glyph> generate_glyphs(ResidentFont& owner, double time);''').encode()
    demo = base._replace(base._prefix(files['src/demo.cpp'].decode(), 'int main('),
                         'native_live_plot_abi_v1.h', Path(ABI).name)
    files['src/demo.cpp'] = (demo + _template('_demo.cpp.in')).encode()
    files[ABI] = (_TEMPLATES / Path(ABI).name).read_bytes()
    expected = set(report['linkage']['private'])
    direct = [name for name in report['linkage']['library']['needed']
              if name.startswith(('libharfbuzz-', 'libfreetype-'))]
    _require(len(direct) == 2 and expected == {name.removeprefix('lib/') for name in files if name.startswith('lib/')},
             'resident package must preserve the native provider closure')
    return report, files, direct, expected


def _measure(root, cases):
    _require(len(cases) >= 2 and all(len(row['roots']) == 3 and row['roots'][:2] == cases[0]['roots'][:2]
                                    for row in cases), 'require successive live states with exact invariant coordinates')

    def arguments(descriptor):
        reference = descriptor['input']
        read_artifact(reference, json_document=False)
        return [reference['uri'], str(descriptor['offset'])]

    with TemporaryDirectory(prefix='scar-resident-plot-trial-') as temporary:
        out = Path(temporary)
        bad = out / 'wrong.ctt'
        layout = bytearray((root / LAYOUT).read_bytes())
        layout[-1] ^= 1
        bad.write_bytes(layout)
        symlink = out / 'symlink.ctt'
        symlink.symlink_to(root / LAYOUT)
        stale = out / 'stale.png'
        run = [root / base.DEMO, root / LAYOUT, root / FONT, bad, symlink, stale,
               *arguments(cases[0]['roots'][0]), *arguments(cases[0]['roots'][1])]
        outputs = []
        for i, row in enumerate(cases):
            output = out / f'frame-{i}.png'
            outputs.append(output)
            run.extend((output, row['time'], *arguments(row['roots'][2])))
        subprocess.run(run, env={'LC_ALL': 'C'}, check=True, timeout=40)
        _require(not stale.exists(), 'a destroyed render owner published an output')
        for i, (output, row) in enumerate(zip(outputs, cases, strict=True)):
            _require(output.read_bytes() == read_artifact(row['expected'], json_document=False),
                     f'native resident PNG differs from original output for case {i}')


def _report(root, reference, origin, files, linkage):
    names = (*files, LIBRARY, base.DEMO, 'CMakeLists.txt')
    return dict(schema=SCHEMA, schema_version=1, source_provider=reference, sources=origin['sources'],
                profile=origin['profile'], validation=origin['validation'],
                files={name: {**base._ref(root / name, 'scar.native-live-plot-member'), 'uri': name} for name in names},
                linkage=linkage, cxx_standard=20, abi_version=2, symbols=SYMBOLS, library=LIBRARY, layout=LAYOUT, font=FONT,
                in_process=True, python_runtime_required=False, exact_original_png_bytes=True,
                immutable_snapshots=['LAYOUT', 'FONT', 'X', 'Y'], frame_inputs=['Z', 'TIME', 'OUTPUT_PATH'],
                single_callback_thread=True, reuses_font_face_and_z_buffer=True,
                mutated_or_symlink_asset_rejected_before_owner_creation=True, stale_owner_rejected=True,
                whole_entry_authorized=False, deployment_authorized=False)


def export(provider, out):
    provider = Path(provider).resolve(strict=True)
    reference = base._ref(provider / 'result.json', base.SCHEMA)
    original, files, direct, expected = _prepare(reference)
    package = Path(out).absolute()
    _require(not package.exists() and not package.is_symlink(), 'require a fresh package path')
    with TemporaryDirectory(prefix='.scar-live-plot-resident-', dir=package.parent) as temporary:
        root = Path(temporary) / 'package'
        for name, payload in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        base._project(root, files, direct)
        base._build(root)
        linkage = base._linkage(root, expected)
        _measure(root, original['validation'])
        report = _report(root, reference, original, files, linkage)
        (root / 'result.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
        root.rename(package)
    return report


def check(package):
    root = Path(package).resolve(strict=True)
    report = read_artifact(base._ref(root / 'result.json', SCHEMA))
    reference = report['source_provider']
    original, files, direct, expected = _prepare(reference)
    linkage = base._linkage(root, expected)
    _require(report == _report(root, reference, original, files, linkage), 'resident provider receipt differs')
    for name, payload in files.items():
        _require(base._member(root, report, name) == payload, 'resident source or provider differs: ' + name)
    with TemporaryDirectory(prefix='scar-resident-plot-project-') as temporary:
        draft = Path(temporary)
        base._project(draft, files, direct)
        _require((root / 'CMakeLists.txt').read_bytes() == (draft / 'CMakeLists.txt').read_bytes(),
                 'resident project metadata differs')
    _measure(root, original['validation'])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider')
    parser.add_argument('--out')
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check and not any((args.provider, args.out)):
        result = check(args.check)
    elif not args.check and args.provider and args.out:
        result = export(args.provider, args.out)
    else:
        parser.error('provide --check PACKAGE or --provider PACKAGE --out NEW_PACKAGE')
    print(json.dumps(dict(frames=len(result['validation']), same_png_bytes=result['exact_original_png_bytes'],
                          stale_owner_rejected=result['stale_owner_rejected'], symbols=result['symbols'])))


if __name__ == '__main__':
    main()

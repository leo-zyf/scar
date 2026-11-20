"""Compile diagnostic phase clocks against two unchanged native plot releases."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from scar.analysis import native_plot_phases as analysis
from scar.codegen import native_live_plot as base
from scar.codegen import native_live_plot_resident as resident
from scar.ir.execution_recipe import read_artifact

SCHEMA = 'scar.native-live-plot-phase'
ROW_SCHEMA = analysis.SAMPLE
PHASES = analysis.PHASES
PNG_DETAIL_PHASES = analysis.PNG_DETAIL_PHASES
COUNTERS = tuple(f'{phase}_{kind}' for phase in PHASES for kind in ('ns', 'count'))
TSV_COLUMNS = ('backend', 'sequence', 'operation', 'frame', 'wall_ns', *COUNTERS)
_TEMPLATES = Path(__file__).parent
_INCLUDE = '#include "native_live_plot_phase.hpp"\n'
_HEADER = 'include/scar_generated/native_live_plot_phase.hpp'


def _require(condition, detail):
    if not condition:
        raise ValueError('native live plot phase: ' + detail)


def _template(suffix):
    return (_TEMPLATES / ('native_live_plot_phase' + suffix)).read_text()


def _body(source, marker, declaration):
    _require(source.count(marker) == 1, 'provider boundary differs: ' + marker)
    start = source.index('{', source.index(marker) + len(marker)) + 1
    return source[:start] + '\n    ' + declaration + source[start:]


def _scope(source, marker, phase):
    return _body(source, marker, f'const scar::plot_phase::Scope phase(scar::plot_phase::Phase::{phase});')


def _observe(source, expression, phase):
    indent = expression[:len(expression) - len(expression.lstrip())]
    return base._replace(source, expression,
        indent + f'scar::plot_phase::observe(scar::plot_phase::Phase::{phase}, [&] {{\n' +
        '\n'.join('    ' + row for row in expression.splitlines()) + '\n' + indent + '});')


def _png_detail(files):
    path = 'vendor/pillow/ZipEncode.c'
    source = files[path].decode()
    enter = '                    if (context->mode == ZIP_PNG) {'
    source = base._replace(source, enter, enter + '\n'
        '                        const uint64_t scar_filter_start = scar_plot_png_filter_begin_v2();')
    exit_filter = '                    }\n\n                    /* Compress this line */'
    source = base._replace(source, exit_filter,
        '                        scar_plot_png_filter_end_v2(scar_filter_start);\n' + exit_filter)
    for old, new, count in (
            ('deflateInit2(', 'scar_plot_png_deflate_init_v2(', 1),
            ('deflateSetDictionary(', 'scar_plot_png_deflate_dictionary_v2(', 1),
            ('deflateEnd(', 'scar_plot_png_deflate_end_v2(', 3),
            ('deflate(&context->z_stream, Z_NO_FLUSH)', 'scar_plot_png_deflate_v2(&context->z_stream, Z_NO_FLUSH)', 2),
            ('deflate(&context->z_stream, Z_FINISH)', 'scar_plot_png_deflate_v2(&context->z_stream, Z_FINISH)', 1)):
        _require(source.count(old) == count, 'original PNG compression call markers differ: ' + old)
        source = source.replace(old, new)
    include = '#include "ZipCodecs.h"\n'
    source = base._replace(source, include, include + '''
#include <stdint.h>
uint64_t scar_plot_png_filter_begin_v2(void);
void scar_plot_png_filter_end_v2(uint64_t);
int scar_plot_png_deflate_v2(z_streamp, int);
int scar_plot_png_deflate_init_v2(z_streamp, int, int, int, int, int);
int scar_plot_png_deflate_dictionary_v2(z_streamp, const Bytef*, uInt);
int scar_plot_png_deflate_end_v2(z_streamp);
''')
    files[path] = source.encode()

    path = 'src/png.cpp'
    source = files[path].decode()
    for call in ('std::fwrite(data, 1, length, file)', '::mkstemp(temporary.data())',
                 '::fdopen(descriptor, "wb")', 'std::fclose(file)'):
        source = base._replace(source, call,
            f'scar::plot_phase::observe(scar::plot_phase::Phase::PngTempIo, [&] {{ return {call}; }})')
    original = '''    const bool published = status == 0 && !write_error && close_error == 0 &&
        ::link(temporary.c_str(), requested.c_str()) == 0;
    ::unlink(temporary.c_str());'''
    source = base._replace(source, original, '''    const bool published = scar::plot_phase::observe(
        scar::plot_phase::Phase::PngFinalPublish, [&] {
            const bool linked = status == 0 && !write_error && close_error == 0 &&
                ::link(temporary.c_str(), requested.c_str()) == 0;
            ::unlink(temporary.c_str());
            return linked;
        });''')
    files[path] = source.encode()


def _instrument_live(raw, variant):
    source = raw.decode()
    source = _scope(source, 'std::vector<char> profile_bytes(', 'AssetReadVerify')
    native = 'extern "C" int scar_live_plot_'
    if variant == 'v1':
        source = _observe(source, '''    selected.x = borrow_lattice(tape, selected, x);
    selected.y = borrow_lattice(tape, selected, y);''', 'CoordinatesPackGuard')
        source = _observe(source, '''    guard_plot_coordinates(tape, grid.x);
    guard_plot_coordinates(tape, grid.y);''', 'CoordinatesPackGuard')
        source = _observe(source, '    selected.z = borrow_lattice(tape, selected, z);', 'ZPack')
        source = _scope(source, 'std::vector<char> compose_scene(', 'ContourScene')
        source = _body(source, native + 'v1(', 'scar::plot_phase::Operation operation("render");')
    else:
        source = _observe(source, '''        copy_lattice(x_view, grid, x, grid.x);
        copy_lattice(y_view, grid, y, grid.y);''', 'CoordinatesPackGuard')
        source = _observe(source, '''        guard_plot_coordinates(tape, grid.x);
        guard_plot_coordinates(tape, grid.y);''', 'CoordinatesPackGuard')
        source = _observe(source, '    copy_lattice(owner.z_view, grid, z, grid.z);', 'ZPack')
        source = _scope(source, 'std::vector<char> compose_resident(', 'ContourScene')
        source = _body(source, native + 'prepare_v2(', 'scar::plot_phase::Operation operation("prepare");')
        source = _body(source, native + 'render_v2(', 'scar::plot_phase::Operation operation("render");')
        source = _body(source, 'extern "C" void scar_live_plot_destroy_v2(',
                       'scar::plot_phase::Operation operation("destroy", context != nullptr);')
        source = base._replace(source, '    owners.erase(reinterpret_cast<std::uintptr_t>(context));',
                               '    owners.erase(reinterpret_cast<std::uintptr_t>(context));\n    operation.success();')
    success = '        report_error(error, capacity, "");\n        return 0;'
    _require(source.count(success) == (1 if variant == 'v1' else 2), 'provider success boundary differs')
    return (_INCLUDE + source.replace(success, '        operation.success();\n' + success)).encode()


def _demo(source, variant):
    preamble = base._prefix(source.decode(), 'int main(' if variant == 'v1' else 'struct Prepared {')
    if variant == 'v1':
        options = dict(OWNER='', PREPARE='', DESTROY='',
            RENDER='scar_live_plot_v1(&x, &y, &z, frame.time, argv[1], argv[2], output.c_str(), '
                   'detail.data(), detail.size())')
    else:
        options = dict(OWNER='''struct Owner {
    void* value = nullptr;
    ~Owner() { if (value) scar_live_plot_destroy_v2(value); }
};''', PREPARE='''            Owner owner;
            timed(trace.get(), sequence, "prepare", -1, [&] {
                if (scar_live_plot_prepare_v2(&x, &y, argv[1], argv[2], &owner.value, detail.data(), detail.size()))
                    throw std::runtime_error(detail.data());
            });''', DESTROY='''            timed(trace.get(), sequence, "destroy", -1, [&] {
                scar_live_plot_destroy_v2(std::exchange(owner.value, nullptr));
            });''', RENDER='scar_live_plot_render_v2(owner.value, &z, frame.time, output.c_str(), '
                              'detail.data(), detail.size())')
    template = _template('_demo.cpp.in')
    for name, value in options.items():
        template = base._replace(template, '@' + name + '@', value)
    return (preamble + template).encode()


def _staged(root, original, variant, png_detail=False):
    files = {name: base._member(root, original, name) for name in original['files']
             if name not in (base.LIBRARY, base.DEMO, 'CMakeLists.txt')}
    files['src/live.cpp'] = _instrument_live(files['src/live.cpp'], variant)
    source = _scope(files['src/glyph.cpp'].decode(), 'explicit Font(', 'FontSetup')
    files['src/glyph.cpp'] = (_INCLUDE + _scope(source, 'std::vector<Glyph> generate_glyphs(', 'Glyphs')).encode()
    for name, signature, phase in (('agg', 'void scar::live::render_scene(', 'AggRaster'),
                                   ('png', 'void scar::live::save_png(', 'PngEncodePublish')):
        path = f'src/{name}.cpp'
        files[path] = (_INCLUDE + _scope(files[path].decode(), signature, phase)).encode()
    if png_detail:
        _png_detail(files)
    files['src/demo.cpp'] = _demo(files['src/demo.cpp'], variant)
    files['src/phase.cpp'] = _template('.cpp.in').encode()
    header = base._replace(_template('.hpp.in'), '@BACKEND@', variant)
    files[_HEADER] = base._replace(header, '@PNG_DETAIL@', str(int(png_detail))).encode()
    direct = [name for name in original['linkage']['library']['needed']
              if name.startswith(('libharfbuzz-', 'libfreetype-'))]
    _require(len(direct) == 2, 'original native font dependencies differ')
    return files, direct


def _phases(png_detail):
    return analysis.phases(1 + int(png_detail))


def _columns(png_detail):
    if not png_detail:
        return TSV_COLUMNS
    return TSV_COLUMNS + tuple(f'{phase}_{kind}' for phase in PNG_DETAIL_PHASES for kind in ('ns', 'count'))


def _rows(tsv, native_log, variant, count, *, png_detail=False, png_counts=()):
    phases, columns = _phases(png_detail), _columns(png_detail)
    _require(not png_detail or len(png_counts) == count, 'PNG oracle cadence is absent')
    with tsv.open(newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        _require(tuple(reader.fieldnames) == columns, 'phase trace field names differ')
        rows = [{key: int(value) if key not in ('backend', 'operation') else value
                 for key, value in row.items()} for row in reader]
    logs = [json.loads(line) for line in native_log.read_text().splitlines()]
    expected = [('render', i) for i in range(count)]
    if variant == 'v2':
        expected = [('prepare', -1), *expected, ('destroy', -1)]
    _require([(row['operation'], row['frame']) for row in rows] == expected and len(logs) == len(rows),
             'phase rows do not represent one complete native frame sequence')
    for row, record in zip(rows, logs, strict=True):
        started = row['operation']
        expected_counts = dict.fromkeys(phases, 0)
        if started == 'prepare' or (started == 'render' and variant == 'v1'):
            expected_counts.update(asset_read_verify=2, font_setup=1, coordinates_pack_guard=2)
        if started == 'render':
            expected_counts.update(glyphs=1, z_pack=1, contour_scene=1, agg_raster=1, png_encode_publish=1)
            if png_detail:
                height, temp_calls = png_counts[row['frame']]
                _require(row['png_deflate_count'] >= height + 1,
                         'PNG compression did not account for feeding and flushing the original row sequence')
                expected_counts.update(png_filter_selection=height, png_temp_io=temp_calls, png_zlib_lifecycle=2,
                                       png_final_publish=1, png_deflate=row['png_deflate_count'])
        total = sum(row[phase + '_ns'] for phase in phases)
        _require(record['schema'] == ROW_SCHEMA and record['schema_version'] == 1 + int(png_detail) and
                 record['success'] is True and row['backend'] == variant and row['sequence'] == 0 and
                 all(row[key] == record[key] for key in columns if key != 'wall_ns') and
                 0 <= total <= record['wall_ns'] <= row['wall_ns'] and
                 all(row[name + '_count'] == expected and row[name + '_ns'] >= 0 and
                     (expected > 0 or row[name + '_ns'] == 0) for name, expected in expected_counts.items()),
                 'measured phase nesting or released boundary cadence differs')
    return rows, logs


def _validate(root, original, variant, png_detail=False):
    cases = original['validation']
    _require(len(cases) >= 2 and all(len(row['roots']) == 3 and row['roots'][:2] == cases[0]['roots'][:2]
                                    for row in cases), 'original frame sequence does not share coordinates')

    def descriptor(row):
        read_artifact(row['input'], json_document=False)
        return row['input']['uri'], str(row['offset'])

    with TemporaryDirectory(prefix='scar-plot-phase-oracle-') as temporary:
        prefix = Path(temporary)
        tsv, log = prefix / 'phase.tsv', prefix / 'phase.jsonl'
        args = [root / base.DEMO, root / base.LAYOUT, root / base.FONT, tsv, '1',
                *descriptor(cases[0]['roots'][0]), *descriptor(cases[0]['roots'][1])]
        outputs = []
        for i, row in enumerate(cases):
            target = prefix / f'frame-{i}'
            outputs.append(target.with_suffix('.0.png'))
            args.extend((*descriptor(row['roots'][2]), row['time'], target))
        result = subprocess.run(args, env={'LC_ALL': 'C', 'SCAR_PLOT_PHASE_LOG': str(log)}, check=True,
                                capture_output=True, timeout=50)
        _require(not result.stdout and not result.stderr, 'diagnostic changed observable output')
        png_counts = []
        for output, row in zip(outputs, cases, strict=True):
            expected = read_artifact(row['expected'], json_document=False)
            _require(output.read_bytes() == expected,
                     'diagnostic PNG differs from the original witness')
            if png_detail:
                png_counts.append(analysis.png_call_counts(expected))
        _rows(tsv, log, variant, len(cases), png_detail=png_detail, png_counts=png_counts)
        target = root / 'validation'
        (target / 'oracle.tsv').write_bytes(tsv.read_bytes())
        (target / 'oracle.jsonl').write_bytes(log.read_bytes())
    return cases


def _arm(source, original, root, variant, png_detail=False):
    source_ref = base._ref(source / 'result.json', original['schema'])
    _require(original == read_artifact(source_ref), 'released provider receipt changed')
    files, direct = _staged(source, original, variant, png_detail)
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    base._project(root, files, direct)
    base._build(root)
    linkage = base._linkage(root, set(original['linkage']['private']))
    loader = 'ld-linux-x86-64.so.2'
    def without_loader(table):
        return {**table, 'needed': [name for name in table['needed'] if name != loader]}
    _require(without_loader(linkage['library']) == without_loader(original['linkage']['library']) and
             linkage['private'] == original['linkage']['private'] and linkage['system'] == original['linkage']['system'],
             f'provider dependency closure changed in diagnostic: {linkage["library"]}')
    validation = _validate(root, original, variant, png_detail)
    names = (*files, base.LIBRARY, base.DEMO, 'CMakeLists.txt', 'validation/oracle.tsv', 'validation/oracle.jsonl')
    products = {name: {**base._ref(root / name, 'scar.plot-phase-member'), 'uri': f'{variant}/{name}'}
                for name in names}
    return dict(original_provider=source_ref, diagnostic_library=products[base.LIBRARY],
                library=f'{variant}/{base.LIBRARY}', demo=f'{variant}/{base.DEMO}',
                layout=f'{variant}/{base.LAYOUT}', font=f'{variant}/{base.FONT}',
                files=products, linkage=linkage, validation=validation)


def export(v1, v2, out, *, png_detail=False):
    sources = [Path(value).resolve(strict=True) for value in (v1, v2)]
    originals = [base.check(sources[0]), resident.check(sources[1])]
    _require(originals[1]['source_provider'] == base._ref(sources[0] / 'result.json', base.SCHEMA) and
             originals[0]['validation'] == originals[1]['validation'], 'diagnostic requires identically scoped releases')
    package = Path(out).absolute()
    _require(not package.exists() and not package.is_symlink(), 'require a fresh diagnostics path')
    with TemporaryDirectory(prefix='.scar-plot-phase-', dir=package.parent) as temporary:
        root = Path(temporary) / 'package'
        root.mkdir()
        variants = ('v1', 'v2')
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {key: executor.submit(_arm, source, original, root / key, key, png_detail)
                       for key, source, original in zip(variants, sources, originals, strict=True)}
            arms = {key: item.result() for key, item in futures.items()}
        report = dict(schema=SCHEMA, schema_version=1, arms=arms, phases=_phases(png_detail), units='ns',
                      clock='CXX_STEADY_CLOCK', phase_accounting='EXCLUSIVE_WITH_NESTED_SCOPES_SUBTRACTED',
                      tsv_columns=_columns(png_detail), native_log_env='SCAR_PLOT_PHASE_LOG', native_log_schema=ROW_SCHEMA,
                      row_schema_version=1 + int(png_detail),
                      all_oracle_pngs_exact=True, diagnostics_only=True, deployment_authorized=False,
                      excludes_fixture_ingress_from_phase_intervals=True)
        (root / 'result.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
        root.rename(package)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('v1', 'v2', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--png-detail', action='store_true')
    args = parser.parse_args(argv)
    report = export(args.v1, args.v2, args.out, png_detail=args.png_detail)
    print(json.dumps(dict(arms=list(report['arms']), phases=report['phases'],
                          all_oracle_pngs_exact=report['all_oracle_pngs_exact'])))


if __name__ == '__main__':
    main()

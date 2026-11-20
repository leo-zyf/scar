"""Lower source-bound rendering and console calls to native controller effects."""

from dataclasses import dataclass
from pathlib import Path
import re

from scar.codegen.native_controller_child import _literal
from scar.ir.execution_recipe import read_artifact


HEADER = 'include/scar_generated/native_controller_plot.hpp'
ABI = 'include/scar_generated/native_live_plot_abi_v1.h'
RESIDENT_ABI = 'include/scar_generated/native_live_plot_resident_abi_v2.h'
ASSETS = Path('assets/plot')
_SOURCE = Path(__file__).with_name('native_controller_plot_v1.hpp.in')
_RESIDENT = Path(__file__).with_name('native_controller_plot_resident_v2.hpp.in')


def _need(condition, reason):
    if not condition:
        raise ValueError('native controller plot: ' + reason)


def _one(items, reason):
    found = list(items)
    _need(len(found) == 1, reason)
    return found[0]


def _exact_slot(source, plan):
    plot = _one((row for row in plan['effect_sites'] if row['site'] == source['effect']['site']),
                'plot does not bind one original effect site')
    formals = source['effect']['formal_routes']
    args = [row['argument_ordinal'] for row in formals.values() if row.get('kind') == 'CONTROLLER_SCALAR']
    time = source['title']['controller_scalar']['argument_ordinal']
    _need(sorted(args) == [0, 1] and time in args and plot['positional_count'] == 2 and not plot['keywords'],
          'exact original two-scalar snapshot calling convention required')
    _need(source['control']['binding']['route'] == 'INITIAL_DEFAULT'
          and type(source['control']['required']) is bool,
          'snapshot branch must come from its exact callable default')
    return plot, time, 1 - time


def _coords(bindings, startup):
    offsets = {}
    for axis in ('x', 'y'):
        route = bindings[axis]
        found = _one((row for row in startup.auxiliary_arrays if row['id'] == route['id']),
                     'source plot coordinate root is missing from the startup image')
        _need(route['kind'] == 'STARTUP_ASSET' and found['shape'] == route['root_shape']
              and found['path'] == route['controller_path'],
              'startup array and original coordinate view differ')
        offsets[axis] = found
    return offsets


def _provider_files(manifest, root, *, resident=False):
    files = manifest['files']
    explicit = {manifest[key] for key in ('library', 'layout', 'font')}
    names = explicit | {name for name in files if name.startswith(('lib/', 'licenses/'))}
    _need(explicit <= files.keys(),
          'provider needs its exact executable and read-only assets')
    package, guards = {}, []
    for name in sorted(names):
        reference = files[name]
        path = Path(name)
        _need(reference['uri'] == name and not path.is_absolute() and '..' not in path.parts,
              'provider member needs a package-relative path')
        out = str(ASSETS / path)
        package[out] = reference['schema'], read_artifact(
            {**reference, 'uri': str(root / path)}, json_document=False)
        if not name.startswith('licenses/') and (not resident or name not in (manifest['layout'], manifest['font'])):
            guards.append('{' + f'{_literal(out)}, {reference["size_bytes"]}, "{reference["sha256"]}"' + '}')
    if resident:
        abi = files[RESIDENT_ABI]
        _need(abi['uri'] == RESIDENT_ABI, 'resident ABI header moved')
        package[RESIDENT_ABI] = abi['schema'], read_artifact(
            {**abi, 'uri': str(root / RESIDENT_ABI)}, json_document=False)
    return package, guards


def _template(resident):
    source = _SOURCE.read_text()
    if not resident:
        return source
    from scar.codegen.native_live_plot import _between, _replace

    source = _replace(source, '#include "native_live_plot_abi_v1.h"',
                      '#include "native_live_plot_resident_abi_v2.h"')
    source = _between(source, '    class Library {', '    const scar_array_v1 x_, y_;',
                      _RESIDENT.read_text() + '\n')
    source = _replace(source, '    std::filesystem::path layout_, font_;\n', '')
    source = _replace(source, '''        const auto result = render_(&x_, &y_, &roots[@Z_ROOT@], time, layout_.c_str(), font_.c_str(),
                                    path.c_str(), error.data(), error.size());''',
                      '''        const auto result = library_.render(roots[@Z_ROOT@], time, path.c_str(),
                                            error.data(), error.size());''')
    return _replace(source, '''        layout_ = package / @LAYOUT@;
        font_ = package / @FONT@;
        render_ = library_.load((package / @LIBRARY@).c_str());''',
                    '''        const auto layout = package / @LAYOUT@;
        const auto font = package / @FONT@;
        library_.load((package / @LIBRARY@).c_str(), x_, y_, layout.c_str(), font.c_str());''')


@dataclass(frozen=True, slots=True)
class NativeControllerPlot:
    reference: dict
    provider: dict
    binding: dict
    source: bytes
    files: dict
    residency: dict | None = None

    def members(self):
        members = {HEADER: ('scar.cpp-header', self.source), **self.files}
        if self.residency is None:
            members[ABI] = ('scar.cpp-header', Path(__file__).with_name('native_live_plot_abi_v1.h').read_bytes())
        return members

    def artifacts(self):
        report = dict(planning=self.reference, provider=self.provider, binding=self.binding,
                      materializes_runtime_paths_and_pixels=True, borrows_current_controller_roots=True,
                      synchronous=True, external_effect_library_required=False, whole_entry_authorized=False,
                      deployment_authorized=False, python_runtime_required=False)
        if self.residency is not None:
            report['residency'] = self.residency
        return report


def prepare(reference, provider_ref, *, controller_plan, startup, console, child, ports, residency=None):
    from scar.analysis.source_controller_plot import check as check_plot

    if residency is None:
        from scar.codegen.native_live_plot import check as check_provider
    else:
        from scar.analysis.source_controller_plot_residency import check as check_residency
        from scar.codegen.native_live_plot_resident import check as check_provider

    source = check_plot(reference['uri'])
    _need(source == read_artifact(reference) and source['startup'] == startup.reference,
          'plot source and startup identities differ')
    root = Path(provider_ref['uri']).parent
    provider = check_provider(root)
    _need(provider == read_artifact(provider_ref) and provider['python_runtime_required'] is False,
          'native provider receipt differs')
    if residency is not None:
        admitted = check_residency(residency['uri'])
        _need(admitted == read_artifact(residency) and admitted['source_plot'] == reference
              and admitted['source_provider'] == provider['source_provider'] and provider['abi_version'] == 2
              and admitted['legality'] == 'CONDITIONAL_EXPLICIT_TRIAL_WITH_IMMUTABLE_COMPILER_OWNED_PACKAGE',
              'resident provider must implement the exact admitted plot and stateless provider')
    layout = read_artifact(provider['sources']['layout'])
    _need(layout['contour_plan'] == source['contour'] and layout['title_plan'] == source['text'],
          'renderer must lower the exact original contour and title bindings')
    plot_site, time_idx, step_idx = _exact_slot(source, controller_plan)
    sites = {plot_site['ordinal'], console.site['ordinal'], child.site['ordinal']}
    _need(len(sites) == 3 and sites == {row['ordinal'] for row in controller_plan['effect_sites']},
          'the native callbacks must cover every original controller effect')
    array_ports = [row for row in ports if row['kind'] == 'array']
    arrays = source['arrays']
    roots = _coords(arrays, startup)
    z = arrays['z']
    z_ordinal = _one((idx for idx, row in enumerate(array_ports) if row['id'] == z['id']
                      and row['shape'] == z['root_shape'] and row['output']),
                     'live plot data must borrow the matching updated array port')
    _need(z['kind'] == 'CONTROLLER_ARRAY', 'plot field must get its current numerical state')
    for scene in layout['scenes']:
        for axis, route in scene['original_root_views'].items():
            _need(all(route[name] == arrays[axis][name]
                      for name in ('root_shape', 'start', 'shape', 'byte_strides')),
                  'physical view must match the native renderer layout')
    directory = source['output']['directory']
    filename = source['output']['filename']
    counter = source['counter']
    _need(directory['path'] == child.contract['directory_field_path'],
          'plot and child must share the original directory route')
    _need(type(counter['initial']) is int and 0 <= counter['initial'] < (1 << 63) and counter['increment'] == 1
          and type(filename['minimum_width']) is int and 1 <= filename['minimum_width'] <= 32
          and filename['format'] == f"0{filename['minimum_width']}d"
          and all(re.fullmatch(r'[A-Za-z0-9_.-]*', filename[key]) for key in ('prefix', 'suffix')),
          'unsupported original persistent counter or filename contract')
    package, guards = _provider_files(provider, root, resident=residency is not None)
    replacements = dict(
        ASSET_COUNT=len(guards), ASSETS=', '.join(guards), ROOT_COUNT=len(array_ports),
        INITIAL_COUNTER=counter['initial'], PLOT_ORDINAL=plot_site['ordinal'], PLOT_ARITY=2,
        STEP_ORDINAL=step_idx, TIME_ORDINAL=time_idx, PRINT_ORDINAL=console.site['ordinal'],
        Z_ROOT=z_ordinal, INDEX_WIDTH=filename['minimum_width'],
        PREFIX=_literal(filename['prefix']), SUFFIX=_literal(filename['suffix']),
        **{key.upper(): _literal(str(ASSETS / provider[key])) for key in ('library', 'layout', 'font')},
        **{f'{axis.upper()}_{key}': value
           for axis, row in roots.items() for key, value in zip(('OFFSET', 'ROWS', 'COLS'),
                                                                (row['offset'], *row['shape']), strict=True)},
    )
    code = re.sub(r'@([A-Z_]+)@', lambda match: str(replacements[match[1]]), _template(residency is not None))
    binding = dict(plot_ordinal=plot_site['ordinal'], print_ordinal=console.site['ordinal'],
                   child_ordinal=child.site['ordinal'], time_argument=time_idx, step_argument=step_idx,
                   ordered_filename_counter=filename, read_only_roots=roots, current_root_ordinal=z_ordinal,
                   directory_argument=directory['path'], provider_members=sorted(package))
    if residency is not None:
        binding['provider_lifetime'] = 'ONE_CONTEXT_PER_CONTROLLER_WITH_LIVE_Z_AND_TIME'
    return NativeControllerPlot(reference, provider_ref, binding, code.encode(), package, residency)

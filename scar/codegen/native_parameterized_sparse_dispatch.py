"""Lower the checked exact-count width selection to a relocatable C++20 entry."""

import argparse
import json
import os
from pathlib import Path
import shlex
from shutil import copy2, copytree
from tempfile import TemporaryDirectory

from scar.analysis.replay_transaction import replay_transaction
from scar.codegen import native_parameterized_sparse_entry as entry
from scar.codegen import scalar_recursive_entry as host
from scar.codegen.native_project import build_native_project, write_native_project
from scar.ir.execution_recipe import read_artifact
from scar.optimization import native_parameterized_sparse_width as width_plan
from scar.optimization.scalar_score_ir import _require


SCHEMA = 'scar.native-parameterized-sparse-dispatch'
_PROGRAM = 'scar_sparse_dispatch'
_SOURCE = Path(__file__).with_suffix('.cpp.in')


def _materials(selection_path):
    reference = host._ref(selection_path, width_plan.SCHEMA)
    selection = width_plan.check(reference['uri'])
    default = selection['default_width']
    overrides = sorted((int(count), width) for count, width in selection['exact_count_to_width'].items()
                       if width != default)
    route_widths = sorted({default, *(width for _, width in overrides)})
    source = _SOURCE.read_text().replace('@OVERRIDE_COUNT@', str(len(overrides)))
    source = source.replace('@OVERRIDES@', ', '.join(f'{{{count}, {width}}}' for count, width in overrides))
    source = source.replace('@DEFAULT_WIDTH@', str(default))
    header = entry._HEADER.read_text().replace('@SCAR_MIN@', str(selection['minimum_count']))
    header = header.replace('@SCAR_MAX@', str(selection['maximum_count']))
    return reference, selection, route_widths, {'src/main.cpp': source, 'include/scar_sparse_runtime_count.hpp': header}


def _project(root):
    write_native_project(root, _PROGRAM, include_dirs=('include',), compile_options=entry._FLAGS,
                         install_dirs=(('arms', '.'),), preserve_install_permissions=True)


def _copy_member(source, destination):
    if Path(source).stat().st_size >= 1024**2:
        return entry._link(source, destination)
    return copy2(source, destination)


def _routes(root, selection, widths):
    routes = {}
    for width in widths:
        observed = selection['arms'][str(width)]
        manifest = f'arms/{width}/artifact.json'
        executable = f'arms/{width}/scar_sparse_parameterized'
        arm = host._local(root, manifest, entry.SCHEMA)
        binary = host._local(root, executable, 'scar.elf')
        _require(arm == dict(observed['artifact'], uri=manifest)
                 and binary == dict(observed['executable'], uri=executable),
                 'package route differs from the measured native arm')
        _require(entry.check(root / manifest) == read_artifact(observed['artifact']),
                 'package route failed independent native validation')
        routes[str(width)] = dict(artifact=arm, executable=binary)
    return routes


def _describe(root, reference, selection, routes, members):
    executable = root / 'bin' / _PROGRAM
    return dict(schema=SCHEMA, schema_version=1, selection=reference,
                binary=host._local(root, f'bin/{_PROGRAM}', 'scar.elf'), arms=routes,
                sources=entry._members(root, [*members, 'CMakeLists.txt']), host_linkage=host._elf(executable),
                count_port=dict(minimum=selection['minimum_count'], maximum=selection['maximum_count'],
                                encoding='CANONICAL_ASCII_DECIMAL'),
                exact_count_to_width=selection['exact_count_to_width'], default_width=selection['default_width'],
                cxx_standard=20, python_runtime_required=False, deployment_authorized=False)


@replay_transaction()
def export(selection_path, out):
    reference, selection, widths, members = _materials(selection_path)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    with TemporaryDirectory(prefix='.scar-sparse-dispatch-', dir=out.parent) as temporary:
        root = Path(temporary) / 'package'
        root.mkdir()
        for name, value in members.items():
            child = root / name
            child.parent.mkdir(exist_ok=True)
            child.write_text(value, encoding='utf8')
        for width in widths:
            origin = Path(selection['arms'][str(width)]['artifact']['uri']).parent
            copytree(origin, root / 'arms' / str(width), copy_function=_copy_member)
        routes = _routes(root, selection, widths)
        _project(root)
        build_native_project(root, _PROGRAM, compiler)
        descriptor = _describe(root, reference, selection, routes, members)
        with (root / 'artifact.json').open('x', encoding='utf8') as stream:
            json.dump(descriptor, stream, indent=2, sort_keys=True)
            stream.write('\n')
        root.rename(out)
    return descriptor


@replay_transaction()
def check(path):
    artifact = host._ref(path, SCHEMA)
    descriptor = read_artifact(artifact)
    reference, selection, widths, members = _materials(descriptor['selection']['uri'])
    root = Path(artifact['uri']).parent
    _require(all((root / name).read_bytes() == source.encode('utf8') for name, source in members.items()),
             'sparse dispatch source differs from the checked selection')
    with TemporaryDirectory(prefix='scar-sparse-dispatch-replay-') as temporary:
        replay = Path(temporary)
        _project(replay)
        _require((root / 'CMakeLists.txt').read_bytes() == (replay / 'CMakeLists.txt').read_bytes(),
                 'sparse dispatch project file differs from the checked selection')
    _require(descriptor == _describe(root, reference, selection, _routes(root, selection, widths), members),
             'sparse dispatch manifest differs from independent replay')
    return descriptor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('selection', 'out', 'check'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    if args.check and not any((args.selection, args.out)):
        result = check(args.check)
    elif args.selection and args.out and not args.check:
        result = export(args.selection, args.out)
    else:
        parser.error('provide --selection and --out, or --check ARTIFACT')
    print(json.dumps({key: result[key] for key in ('binary', 'exact_count_to_width', 'default_width')}, sort_keys=True))


if __name__ == '__main__':
    main()

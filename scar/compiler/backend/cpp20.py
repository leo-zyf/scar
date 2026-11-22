"""Emit one C++20 project from a locked native source revision."""

from hashlib import sha256
import json
from pathlib import Path

from scar.codegen.native_cpu_elf import cpu_executable_dynamic
from scar.codegen.native_project import build_native_project, write_native_project

from ..frontend.clang import capture_unit


def _compile_settings(flags, source_root):
    include_dirs = []
    compile_options = []
    tokens = iter(flags)
    for flag in tokens:
        if flag.startswith(('-std=', '--std=')):
            continue
        if flag in ('-std', '--std'):
            if next(tokens, None) is None:
                raise ValueError('missing language standard in native compiler flags')
            continue
        if not flag.startswith('-I'):
            compile_options.append(flag)
            continue
        location = next(tokens, None) if flag == '-I' else flag[2:]
        if not location:
            raise ValueError('missing include directory in native compiler flags')
        include_dir = (source_root / location).resolve()
        if not include_dir.is_relative_to(source_root):
            raise ValueError(f'include directory is outside the locked source root: {location}')
        include_dirs.append((Path('src') / include_dir.relative_to(source_root)).as_posix())
    return tuple(include_dirs), tuple(compile_options)


def _lower_pointer_conversions(path, data, records):
    conversions = {}
    for expression in records:
        parent = expression['parent']
        if (parent is None or expression['path'] != path
                or expression['canonical_type'] != 'void *' or not expression['span']):
            continue
        conversion = records[parent]
        if (conversion['kind'] != 'UnexposedExpr' or conversion['path'] != path
                or not conversion['canonical_type'].endswith(' *')
                or conversion['canonical_type'] == 'void *'):
            continue
        start, end = expression['span']
        if not 0 <= start < end <= len(data):
            continue
        target = conversion['type']
        previous = conversions.setdefault((start, end), target)
        if previous != target:
            raise ValueError(f'ambiguous C pointer conversion in {path} at byte {start}')

    chunks = []
    position = 0
    provenance = []
    for (start, end), target in sorted(conversions.items()):
        if start < position:
            raise ValueError(f'overlapping C pointer conversions in {path} at byte {start}')
        chunks.extend((data[position:start], f'static_cast<{target}>('.encode(),
                       data[start:end], b')'))
        position = end
        provenance.append({'span': [start, end], 'from': 'void *', 'to': target})
    chunks.append(data[position:])
    return b''.join(chunks), provenance


def emit_sources(manifest, project, edits):
    """Stage verified headers and source files, then apply language bridges."""
    if manifest['language'] == 'fortran':
        from .fortran import emit_fortran
        return emit_fortran(manifest, project, edits)

    root = Path(manifest['source_root']).resolve()
    is_c = manifest['language'] == 'c'
    units = {unit['path']: unit for unit in manifest['units']}
    material = manifest['source_lock']['files']
    if not units or not units.keys() <= material.keys():
        raise ValueError('every translation unit needs locked source bytes')
    if not edits.keys() <= units.keys() or any(not isinstance(value, bytes) for value in edits.values()):
        raise ValueError('source edits must provide byte replacements for translation units')

    settings = {_compile_settings(unit['flags'], root) for unit in units.values()}
    if len(settings) != 1:
        raise ValueError('C++20 staging requires a shared compile configuration')
    include_dirs, compile_options = settings.pop()
    sources, translation = {}, []
    for name, lock in material.items():
        relative = Path(name)
        original = (root / relative).resolve()
        if relative.is_absolute() or not original.is_relative_to(root) or '..' in relative.parts:
            raise ValueError(f'source path outside the locked root: {name}')
        data = original.read_bytes()
        if len(data) != lock['size_bytes'] or sha256(data).hexdigest() != lock['sha256']:
            raise ValueError(f'locked source changed: {name}')
        data = edits.get(name, data)
        output = Path('src') / relative
        if name in units:
            if is_c:
                evidence = capture_unit(root, units[name], source_override=edits.get(name))
                data, conversions = _lower_pointer_conversions(name, data, evidence['records'])
                output = output.with_suffix('.cpp')
                translation.append(dict(source=name, output=str(output), c_pointer_conversions=conversions))
            sources[name] = str(output)
        target = project / output
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)

    if is_c:
        (project / 'translation.json').write_text(json.dumps(translation, indent=2) + '\n')
    linkage = manifest if is_c else manifest.get('cxx', {})
    return dict(sources=tuple(sources[name] for name in units), include_dirs=include_dirs,
                compile_options=compile_options,
                libraries=linkage.get('libraries', ()), link_options=linkage.get('link_options', ()),
                runtime_dirs=linkage.get('runtime_dirs', ()))


def build_reference_project(manifest, project):
    """Build the source-preserving C++20 route and check its runtime ABI."""
    project.mkdir()
    build = emit_sources(manifest, project, {})
    write_native_project(project, 'scar_entry', **build)
    executable, _ = build_native_project(project, 'scar_entry', ('g++',))
    dynamic = cpu_executable_dynamic(executable)
    for name in dynamic['needed']:
        if any(provider in name for provider in ('libpython', 'libtorch', 'libgfortran')):
            raise ValueError(f'unqualified target dependency: {name}')
    return executable, dynamic

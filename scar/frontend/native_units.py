"""Normalize admitted native units into sealed executable source fragments."""

from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re

from ..contract import Artifact, artifact, json_digest, read_artifact, write_json

_GNU_CFI_HEADER = "ISO_Fortran_binding.h"
_GNU_CFI_ROUTINES = {"CFI_establish", "CFI_is_contiguous", "CFI_section", "CFI_setpointer"}


def _compile_settings(flags, source_root, staged_root):
    include_dirs = []
    # The generated CMake Release profile defines NDEBUG independently of
    # the original source command. Apply the caller's macro flags after it.
    compile_options = ['-UNDEBUG']
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
        include_dirs.append((staged_root / include_dir.relative_to(source_root)).as_posix())
    return tuple(include_dirs), tuple(compile_options)


def _c_family_evidence(unit, captures, source):
    name = unit['path']
    if name not in captures:
        raise ValueError(f'no sealed C-family type evidence for {name}')
    typed = json.loads(read_artifact(captures[name]))
    fingerprint = typed.get('source_sha256')
    if (typed['path'] != name or typed['flags'] != unit.get('flags', [])
            or fingerprint is not None and fingerprint != sha256(source).hexdigest()):
        raise ValueError(f'C type evidence has a different source or profile: {name}')
    return typed


def _c_spans(units, source_bytes, preserve_abi, captures):
    declarations = defaultdict(list)
    conversions = defaultdict(dict)
    definitions = defaultdict(dict)
    cfi_headers = []
    for unit in units:
        name = unit['path']
        typed = _c_family_evidence(unit, captures, source_bytes[name])
        cfi_headers.extend(typed.get('gnu_cfi', ()))
        records = typed['records']
        functions, parameters, bodies = {}, defaultdict(list), {}
        for index, record in enumerate(records):
            path = record['path']
            if path != name and not preserve_abi:
                continue
            if (preserve_abi and record['external_linkage']
                    and record['kind'] in ('FunctionDecl', 'VarDecl')):
                declarations[path].append(record)
            parent = record['parent']
            if record['kind'] == 'FunctionDecl' and record['definition']:
                functions[index] = record
            elif parent in functions:
                if record['kind'] == 'ParmDecl':
                    parameters[parent].append(record)
                elif record['kind'] == 'CompoundStmt':
                    bodies[parent] = record
            if parent is None or record['canonical_type'] != 'void *' or not record['span']:
                continue
            conversion = records[parent]
            if (conversion['kind'] != 'UnexposedExpr' or conversion['path'] != path
                    or not conversion['canonical_type'].endswith(' *')
                    or conversion['canonical_type'] == 'void *'):
                continue
            span = tuple(record['span'])
            target = conversion['type']
            if conversions[path].setdefault(span, target) != target:
                raise ValueError(f'ambiguous C pointer conversion in {path} at byte {span[0]}')
        for index, arguments in parameters.items():
            if index not in bodies:
                continue
            function = functions[index]
            path = function['path']
            if path not in source_bytes:
                raise ValueError(f'C definition has no locked source bytes: {path}')
            header = _c_identifier_definition(source_bytes[path], function, arguments, bodies[index])
            if header is not None:
                span, replacement, types = header
                if definitions[path].setdefault(span, (replacement, types)) != (replacement, types):
                    raise ValueError(f'ambiguous old-style C function definition. {path}:{function["line"]}')
    return declarations, conversions, definitions, cfi_headers


_TRIVIA = re.compile(rb'(?:\s+|/\*[\s\S]*?\*/|//[^\n]*\n)*')
_C_IDENTIFIER = re.compile(rb'[A-Za-z_]\w*')


def _c_identifier_definition(data, function, parameters, body):
    """Normalize a C identifier-list definition when its call ABI is unchanged."""
    token = _C_IDENTIFIER.match(data, function['offset'])
    if token is None:
        return None
    start = _TRIVIA.match(data, token.end()).end()
    if data[start:start + 1] != b'(':
        return None
    cursor = start + 1
    for index, parameter in enumerate(parameters):
        name = _C_IDENTIFIER.match(data, _TRIVIA.match(data, cursor).end())
        if name is None or name[0] != parameter['name'].encode():
            return None
        delimiter = _TRIVIA.match(data, name.end()).end()
        expected = b')' if index + 1 == len(parameters) else b','
        if data[delimiter:delimiter + 1] != expected:
            return None
        cursor = delimiter + 1

    location = f'{function["path"]}:{function["line"]}'
    if (token[0] != function['name'].encode() or not function['span'] or not body['span']
            or body['path'] != function['path']):
        raise ValueError(f'old-style C definition has no original function/body range. {location}')
    body_start = body['span'][0]
    if not function['span'][0] <= start < cursor <= body_start < function['span'][1]:
        raise ValueError(f'old-style C definition has inconsistent source ranges. {location}')

    # Clang's function type already records the default parameter promotions
    # for a definition written with an identifier list. ParmDecl records the
    # local type before those promotions. Only use an ANSI declarator if each
    # has the same type at both sides of the call boundary.
    types = [parameter['canonical_type'] for parameter in parameters]
    _, _, signature = function['canonical_type'].partition('(')
    if not signature.endswith(')') or signature[:-1].split(', ') != types:
        raise ValueError(f'old-style C parameter promotion or adjustment requires an ABI bridge. {location}')

    declared = []
    names_end = cursor
    for parameter in parameters:
        span = parameter['span']
        if not span or parameter['path'] != function['path']:
            raise ValueError(f'old-style C parameter has no original declaration range. {location}')
        begin, stop = span
        spelling = parameter['name'].encode()
        if start < begin < stop < names_end and data[begin:stop] == spelling and parameter['type'] == 'int':
            declared.append(b'int ' + spelling)
            continue
        name = _C_IDENTIFIER.match(data, parameter['offset'])
        end = _TRIVIA.match(data, stop).end()
        if (not name or name[0] != spelling or not cursor <= begin <= name.start() < name.end() <= stop <= end < body_start
                or _TRIVIA.match(data, cursor).end() != begin or data[end:end + 1] != b';'):
            raise ValueError(f'old-style C parameters need independent ordered declaration ranges. {location}')
        declared.append(data[begin:stop])
        cursor = end + 1
    if _TRIVIA.match(data, cursor).end() != body_start or data[body_start:body_start + 1] != b'{':
        raise ValueError(f'old-style C definition has an unsupported parameter preamble. {location}')
    replacement = b'(' + b', '.join(declared) + b')'
    replacement += b'\n' * (data[start:body_start].count(b'\n') - replacement.count(b'\n'))
    return (start, body_start), replacement, types


def _c_abi_spans(path, data, declarations):
    # Clang can emit one VarDecl per declarator in a comma-separated declaration.
    groups = {}
    for record in declarations:
        span = record['span']
        offset = record['offset']
        spelling = _C_IDENTIFIER.match(data, offset)
        # The token can be a macro alias. Use the cursor's full declaration
        # extent, then qualify the resulting external symbol at link time.
        if (not span or not spelling or not span[0] <= offset < spelling.end() <= span[1] <= len(data)):
            raise ValueError(f'C linkage declaration lacks a source range. {path}:{record["line"]}')
        start, stop = span
        body = record['kind'] == 'FunctionDecl' and record['definition']
        if start not in groups or stop > groups[start][0]:
            groups[start] = stop, body
    for start, (stop, body) in sorted(groups.items()):
        if body:
            if data[stop - 1:stop] != b'}':
                raise ValueError(f'unanchored C function body. {path}:{start}')
        elif data[stop - 1:stop] != b';':
            end = _TRIVIA.match(data, stop).end()
            if data[end:end + 1] != b';':
                raise ValueError(f'unanchored C declaration end. {path}:{start}')
            stop = end + 1
        yield start, stop


def _lower_c(path, data, declarations, conversions, definitions, regions):
    linkage = list(_c_abi_spans(path, data, declarations))
    insertions = defaultdict(list)
    prior = 0
    for start, stop in linkage:
        if start < prior:
            raise ValueError(f'overlapping C linkage declarations in {path}')
        insertions[start].append(b'extern "C" { ')
        insertions[stop].append(b' }')
        prior = stop
    prior = 0
    provenance = []
    for (start, stop), target in sorted(conversions.items()):
        if not 0 <= start < stop <= len(data):
            continue
        if start < prior:
            raise ValueError(f'overlapping C pointer conversions in {path} at byte {start}')
        insertions[start].append(f'static_cast<{target}>('.encode())
        insertions[stop].insert(0, b')')
        prior = stop
        provenance.append({'span': [start, stop], 'from': 'void *', 'to': target})
    replacements = {start: (stop, text) for (start, stop), (text, _) in definitions.items()}
    converted = _apply_spans(path, data, insertions, replacements)
    return converted, provenance, linkage, _map_body_spans(data, converted, insertions, replacements, regions)


def _map_body_spans(original, converted, insertions, replacements, regions):
    result = {}
    for region in regions:
        start, stop = region['source']['span']
        if any(start < offset < stop for offset in insertions):
            continue
        if any(left < stop and start < right for left, (right, _) in replacements.items()):
            continue
        delta = sum(len(data) - (right - left) for left, (right, data) in replacements.items() if right <= start)
        begin = start + delta + sum(map(len, (data for offset, rows in insertions.items()
                                            if offset <= start for data in rows)))
        end = stop + delta + sum(map(len, (data for offset, rows in insertions.items()
                                          if offset < stop for data in rows)))
        if original[start:stop] == converted[begin:end]:
            result[region['region']] = [begin, end]
    return result


def _apply_spans(path, data, insertions, replacements):
    chunks = []
    position = 0
    for offset in sorted(insertions.keys() | replacements.keys()):
        if offset < position:
            raise ValueError(f'overlapping C source edits in {path} at byte {offset}')
        chunks.extend((data[position:offset], *insertions.get(offset, ())))
        position = offset
        if offset in replacements:
            position, replacement = replacements[offset]
            chunks.append(replacement)
    chunks.append(data[position:])
    return b''.join(chunks)


def _locked_source(root, name, lock):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root) or '..' in Path(name).parts:
        raise ValueError(f'source path outside the locked root: {name}')
    data = path.read_bytes()
    size = lock.get('size_bytes', lock.get('size'))
    if len(data) != size or sha256(data).hexdigest() != lock['sha256']:
        raise ValueError(f'locked source changed: {name}')
    return data


def _stage(project, name, data):
    target = project / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(data)


def _stage_gnu_cfi(header, project, headers):
    if header is not None:
        origin, digest = header['path'], header['sha256']
        data = read_artifact(header)
    elif headers:
        origin, digest = headers[0]['path'], headers[0]['sha256']
        data = Path(origin).read_bytes()
        if sha256(data).hexdigest() != digest:
            raise ValueError('the GNU CFI binding header changed after type capture')
    else:
        raise ValueError('the Fortran plan must carry its GNU CFI binding header')
    for record in headers:
        if (record['path'], record['sha256']) != (origin, digest):
            raise ValueError('the native call was typed against a different GNU CFI binding header')
    symbols = {symbol for record in headers for symbol in record['symbols']}
    unsupported = symbols - _GNU_CFI_ROUTINES
    if unsupported:
        raise ValueError(f'unsupported standalone GNU CFI routines: {sorted(unsupported)}')

    # It remains a toolchain header when sealed into the relocatable project.
    _stage(project, 'include/scar_generated/' + _GNU_CFI_HEADER, b'#pragma GCC system_header\n' + data)
    return bool(symbols)


def _stage_native_sources(manifest, evidence, project, regions):
    """Translate and stage pinned source inputs using captured type evidence."""
    profile = manifest['language']
    units = {unit['path']: unit for unit in manifest['units']}
    languages = {name: unit.get('language', profile) for name, unit in units.items()}
    material = manifest['source_lock']['files']
    if not units or len(units) != len(manifest['units']) or not units.keys() <= material.keys():
        raise ValueError('every translation unit needs locked source bytes')
    if set(languages.values()) - {'c', 'c++', 'fortran'}:
        raise ValueError('the C++20 backend requires C, C++ or Fortran source units')
    mixed = len(set(languages.values())) > 1
    root = Path(manifest['source_root']).resolve()
    fortran = [units[name] for name, language in languages.items() if language == 'fortran']
    source_bytes = {name: _locked_source(root, name, lock) for name, lock in material.items()}
    captures = evidence.get('compiler_types', {})
    project.mkdir()
    if not mixed and fortran:
        from .fortran_bridge import emit_fortran

        build = emit_fortran(manifest, project, source_bytes, captures=captures)
        build['libraries'] = (*manifest.get('libraries', ()), *build['libraries'])
        for key in ('link_options', 'runtime_dirs'):
            if manifest.get(key):
                build[key] = manifest[key]
        return build, {}
    c_units = [units[name] for name, language in languages.items() if language == 'c']
    cxx_units = [units[name] for name, language in languages.items() if language == 'c++' and fortran]
    c_root = Path('src/c-abi') if mixed else Path('src')
    cxx_root = Path('src/cxx') if mixed else Path('src')
    decls, casts, headers, cfi_headers = _c_spans(c_units, source_bytes, mixed, captures)
    for unit in cxx_units:
        typed = _c_family_evidence(unit, captures, source_bytes[unit['path']])
        cfi_headers.extend(typed.get('gnu_cfi', ()))
    if (decls.keys() | casts.keys() | headers.keys()) - material.keys():
        raise ValueError('C translation referenced project files missing from the source lock')

    settings, translation, anchors = {}, [], {}
    for name, data in source_bytes.items():
        language = languages.get(name)
        unit_regions = regions.get(name, ())
        if language == 'c' or (language is None and mixed and c_units):
            output = c_root / name
            if language == 'c':
                output = output.with_suffix('.cpp')
                settings[output.as_posix()] = _compile_settings(units[name]['flags'], root, c_root)
            converted, conversions, linkage, spans = _lower_c(
                name, data, decls[name], casts[name], headers[name], unit_regions)
            _stage(project, output, converted)
            anchors.update({region: (output.as_posix(), span) for region, span in spans.items()})
            if language == 'c' or conversions or linkage:
                row = dict(source=name, output=str(output), c_pointer_conversions=conversions, c_linkage=linkage)
                if headers[name]:
                    row['c_identifier_definitions'] = [dict(span=span, parameter_types=types)
                                                       for span, (_, types) in sorted(headers[name].items())]
                translation.append(row)
        if language in (None, 'c++'):
            output = cxx_root / name
            _stage(project, output, data)
            anchors.update({row['region']: (output.as_posix(), row['source']['span']) for row in unit_regions})
            if language == 'c++':
                settings[output.as_posix()] = _compile_settings(units[name]['flags'], root, cxx_root)

    if translation:
        (project / 'translation.json').write_text(json.dumps(translation, indent=2) + '\n')
    fortran_libraries = ()
    runtime_support = ()
    if fortran:
        from .fortran_bridge import emit_fortran

        config = dict(manifest, units=fortran, language='fortran')
        bridge = emit_fortran(config, project, source_bytes, source_directory='src/fortran-abi',
                              mixed=True, captures=captures)
        settings.update({name: ((), bridge['compile_options']) for name in bridge['sources']})
        fortran_libraries = bridge['libraries']
        if cfi_headers or bridge.get('gnu_cfi_descriptors'):
            if _stage_gnu_cfi(evidence.get('gnu_cfi_header'), project, cfi_headers):
                runtime_support = ('gnu-cfi-v1',)

    profiles = (manifest,) if profile == 'c' else (manifest, manifest.get('cxx', {}))
    linkage = {key: tuple(value for profile in profiles for value in profile.get(key, ()))
               for key in ('libraries', 'link_options', 'runtime_dirs')}
    linkage['libraries'] = (*linkage['libraries'], *fortran_libraries)
    common = set(settings.values())
    if runtime_support:
        common.add(((), ()))
    include_dirs, compile_options = next(iter(common)) if len(common) == 1 else ((), ())
    configuration = dict(sources=tuple(settings), include_dirs=include_dirs, compile_options=compile_options,
                         source_settings=settings if len(common) > 1 else {}, **linkage)
    if runtime_support:
        configuration['runtime_support'] = runtime_support
    return configuration, anchors


def _fortran_fragment(region, export, capture, emission, fragments, compiler_types):
    abi, source = region['fortran_abi'], region['source']
    if (abi['profile'] != 'BIND_C_RANK1_ASSUMED_SIZE' or abi['procedure'] != export['procedure']
            or abi['symbol'] != region['symbol'] or source['sha256'] != export['source_sha256']
            or region['return_type'] != 'void' or export['result'] is not None
            or capture is None or capture.get('types') != compiler_types
            or compiler_types is None or compiler_types['sha256'] != abi['semantic_sha256']
            or capture.get('ssa', {}).get('sha256') != abi['ssa_sha256']):
        return None
    parameters, bindings = [], {}
    for arg in export['arguments']:
        name = arg['name']
        if arg['by_value'] and arg['shape'] is None and arg['c_type'] in {'int', 'double'}:
            parameters.append(dict(name=name, type='i32' if arg['c_type'] == 'int' else 'f64', passing='VALUE'))
        elif not arg['by_value'] and 'lower' in arg and arg['c_type'] == 'double':
            pointer = 'ptr<const f64>' if arg['read_only'] else 'ptr<f64>'
            parameters.append(dict(name=name, type=pointer, lower=arg['lower'], passing='ADDRESS'))
        else:
            return None
        bindings[name] = arg['wrapper_name']
    if (parameters != abi['parameters'] or region['parameters'] != [
            dict(name=row['name'], type=row['type']) for row in parameters]):
        return None
    wrapper = export['wrapper']
    if fragments.get(wrapper['path'], {}).get('sha256') != wrapper['sha256']:
        raise ValueError(f'generated Fortran wrapper changed for {export["symbol"]}')
    translation = dict(schema='scar.fortran-region-bridge', schema_version=1, emission=emission,
                       compiler_types=compiler_types, compiler_ssa=capture['ssa'])
    domain = dict(parameter=abi['bounds']['parameter'], minimum=1, maximum=abi['bounds']['maximum'])
    return dict(wrapper, bindings=bindings, translation=translation,
                authority='VERIFIED_ABI_WRAPPER', count_domain=domain)


def prepare_native_units(target, evidence, out, *, regions=()) -> tuple[Artifact, list[dict]]:
    """Seal the C++20-compatible leaves without lowering the whole execution graph."""
    out = Path(out)
    out.mkdir(parents=True)
    pending = out / 'inputs'
    by_unit = defaultdict(list)
    for region in regions:
        by_unit[region['source']['path']].append(region)
    configuration, anchors = _stage_native_sources(target, evidence, pending, by_unit)
    fragments = {}
    for directory in ('src', 'include'):
        for path in sorted((pending / directory).rglob('*')):
            if path.is_file():
                name = path.relative_to(pending).as_posix()
                fragments[name] = asdict(artifact(path))
    emission = pending / 'fortran-emission.json'
    exports = []
    bridge_exports = {}
    captures = {row['unit']: row for row in evidence.get('units', ()) if row['language'] == 'fortran'}
    if emission.is_file():
        data = emission.read_bytes()
        emission_ref = asdict(Artifact(str(emission.resolve()), sha256(data).hexdigest(), len(data)))
        bridge_exports = {(row['unit'], row['symbol']): row for row in json.loads(data)['c_exports']}
        exports = sorted(row['symbol'] for row in bridge_exports.values())
    retained = []
    for region in regions:
        if region['region'] in anchors:
            path, span = anchors[region['region']]
            fragment = {'path': path, 'span': span, 'sha256': fragments[path]['sha256'],
                        'slice_sha256': region['source']['slice_sha256']}
            retained.append({**region, 'fragment': fragment})
        elif 'fortran_abi' in region and (export := bridge_exports.get((region['source']['path'], region['symbol']))):
            unit = region['source']['path']
            fragment = _fortran_fragment(region, export, captures.get(unit), emission_ref,
                                         fragments, evidence['compiler_types'].get(unit))
            if fragment is not None:
                retained.append({**region, 'fragment': fragment})
    seals = {row['region']: json_digest(row) for row in retained}
    return write_json(out / 'native-units.json', dict(schema='scar.native-units-ir', schema_version=2,
                       configuration=configuration, fragments=fragments, exports=exports, region_seals=seals)), retained

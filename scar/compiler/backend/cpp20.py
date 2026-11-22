"""Emit one C++20 project from a locked native source revision."""

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

from scar.artifacts import file_reference
from scar.codegen.native_cpu_elf import cpu_executable_dynamic
from scar.codegen.native_project import build_native_project, write_native_project
from scar.ir.execution_recipe import read_artifact
from scar.ir.implementation_plan import PLAN_SCHEMA, SCHEMA as PROBLEM_SCHEMA, problem_document, problem_from_dict
from scar.ir.program.codec import SCHEMA as PROGRAM_SCHEMA, loads
from scar.optimization.joint_planner import verify_execution_plan

from ..frontend.clang import capture_unit
from ..middle.plan import copy_source_edits, entry_cpus, entry_problem


def admit_entry(artifacts, entry):
    """Replay the artifact and execution contracts before emitting source."""
    schemas = {
        'request': 'scar.native-entry', 'program': PROGRAM_SCHEMA,
        'graph_evidence': 'scar.native-graph-evidence', 'calls': 'scar.native-call-links',
        'hardware': 'scar.target-hardware-graph', 'simplified': PROGRAM_SCHEMA,
        'rewrites': 'scar.native-rewrites', 'cost': 'scar.native-entry-cost',
        'planning_problem': PROBLEM_SCHEMA, 'execution_plan': PLAN_SCHEMA,
    }
    documents = {}
    for role, schema in schemas.items():
        ref = artifacts[role]
        if ref['schema'] != schema:
            raise ValueError(f'invalid {role} artifact schema')
        if role in {'program', 'graph_evidence', 'calls', 'simplified'}:
            if file_reference(ref['uri'], schema) != ref:
                raise ValueError(f'{role} artifact differs from the plan input')
        else:
            documents[role] = read_artifact(ref)
    observation = documents['cost']
    manifest = documents['request']
    rewrite = documents['rewrites']
    if (rewrite['schema_version'] != 1 or rewrite['program'] != artifacts['program']
            or rewrite['result'] != artifacts['simplified']):
        raise ValueError('rewrite ledger does not bind the admitted input and output graphs')
    edits = rewrite.get('source_edits', [])
    typed_units = {unit['path'] for unit in manifest['units']
                   if unit.get('language', manifest['language']) in ('c', 'fortran')}
    evidence = read_artifact(artifacts['graph_evidence']) if edits or typed_units else None
    if evidence and any(evidence[key] != artifacts[key] for key in ('request', 'program', 'hardware', 'calls')):
        raise ValueError('compiler evidence differs from the admitted frontend outputs')
    compiler_types = {row['unit']: row['artifacts']['types'] for row in (evidence or {}).get('units', ())
                      if row['unit'] in typed_units and 'types' in row['artifacts']}
    if compiler_types.keys() != typed_units:
        raise ValueError('C and Fortran translation require original type evidence for every unit')
    if edits:
        if copy_source_edits(manifest, evidence, rewrite['rules']) != edits:
            raise ValueError('source copy edits differ from frontend and IR evidence')
        simplified = loads(read_artifact(artifacts['simplified'], json_document=False).decode('utf-8'))
        remaining = {op.id.wire for op in simplified.operations}
        regions = {region.id.wire for region in simplified.regions}
        if any(edit['operation'] in remaining or edit['region'] not in regions for edit in edits):
            raise ValueError('source edits do not match the simplified IR')
    problem = problem_from_dict(documents['planning_problem'])
    if (observation['schema_version'] != 1 or observation['request'] != artifacts['request']
            or observation['exit'] != 0 or problem_document(entry_problem(entry, artifacts, observation, edits))
            != documents['planning_problem']):
        raise ValueError('native problem differs from the admitted inputs')
    plan = documents['execution_plan']
    verification = verify_execution_plan(problem, plan)
    option = problem.operations[0].options[0]
    if (not plan['baseline_only'] or plan['transfers'] or plan.get('activities')
            or len(plan['selected']) != 1 or plan['selected'][0]['option'] != option.id):
        raise ValueError('backend cannot lower the selected native schedule')
    cpu = int(next(iter(option.demands)).removeprefix('cpu:'))
    if cpu not in entry_cpus(artifacts['hardware']):
        raise ValueError('scheduled CPU is unavailable')
    root = Path(manifest['source_root'])
    for name, lock in manifest['source_lock']['files'].items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError(f'locked source outside the source root: {name}')
        ref = file_reference(path, 'scar.native-source')
        if any(ref[key] != lock[key] for key in ('sha256', 'size_bytes')):
            raise ValueError(f'locked source changed: {name}')
    delivery = dict(schema='scar.native-delivery', schema_version=1, cpus=[cpu],
                    plan=artifacts['execution_plan'], problem=artifacts['planning_problem'],
                    request=artifacts['request'], selected=plan['selected'],
                    source_edits=edits, compiler_types=compiler_types,
                    lowering=option.lowering, verification=verification)
    return manifest, observation, delivery


def _compile_settings(flags, source_root, staged_root):
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
        include_dirs.append((staged_root / include_dir.relative_to(source_root)).as_posix())
    return tuple(include_dirs), tuple(compile_options)


def _c_spans(units, root, edits, preserve_abi, captures):
    declarations = defaultdict(list)
    conversions = defaultdict(dict)
    for unit in units:
        name = unit['path']
        ref = captures.get(name)
        if ref is not None and name not in edits:
            typed = json.loads(read_artifact(ref, json_document=False))
            if typed['path'] != name or typed['flags'] != unit['flags']:
                raise ValueError(f'C type evidence has a different source or profile: {name}')
        else:
            typed = capture_unit(root, unit, source_override=edits.get(name))
        records = typed['records']
        for record in records:
            path = record['path']
            if path != name and not preserve_abi:
                continue
            if (preserve_abi and record['external_linkage']
                    and record['kind'] in ('FunctionDecl', 'VarDecl')):
                declarations[path].append(record)
            parent = record['parent']
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
    return declarations, conversions


_TRIVIA = re.compile(rb'(?:\s+|/\*[\s\S]*?\*/|//[^\n]*\n)*')
_C_IDENTIFIER = re.compile(rb'[A-Za-z_]\w*')


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


def _lower_c(path, data, declarations, conversions):
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
    chunks = []
    position = 0
    for offset, tokens in sorted(insertions.items()):
        chunks.extend((data[position:offset], *tokens))
        position = offset
    chunks.append(data[position:])
    return b''.join(chunks), provenance, linkage


def _locked_source(root, name, lock):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root) or '..' in Path(name).parts:
        raise ValueError(f'source path outside the locked root: {name}')
    data = path.read_bytes()
    if len(data) != lock['size_bytes'] or sha256(data).hexdigest() != lock['sha256']:
        raise ValueError(f'locked source changed: {name}')
    return data


def _stage(project, name, data):
    target = project / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(data)


def emit_sources(manifest, project, edits, *, captures=None):
    """Translate each locked unit under its language profile, then assemble C++20."""
    profile = manifest['language']
    units = {unit['path']: unit for unit in manifest['units']}
    languages = {name: unit.get('language', profile) for name, unit in units.items()}
    material = manifest['source_lock']['files']
    if not units or not units.keys() <= material.keys():
        raise ValueError('every translation unit needs locked source bytes')
    if not edits.keys() <= units.keys() or any(not isinstance(value, bytes) for value in edits.values()):
        raise ValueError('source edits must provide byte replacements for translation units')
    mixed = len(set(languages.values())) > 1
    root = Path(manifest['source_root']).resolve()
    fortran = [units[name] for name, language in languages.items() if language == 'fortran']
    if not mixed and fortran:
        from .fortran import emit_fortran

        for name, lock in material.items():
            _locked_source(root, name, lock)
        return emit_fortran(manifest, project, edits, captures=captures)

    c_units = [units[name] for name, language in languages.items() if language == 'c']
    c_root = Path('src/c-abi') if mixed else Path('src')
    cxx_root = Path('src/cxx') if mixed else Path('src')
    decls, casts = _c_spans(c_units, root, edits, mixed, captures or {})
    if (decls.keys() | casts.keys()) - material.keys():
        raise ValueError('C translation referenced project files missing from the source lock')

    settings, translation = {}, []
    for name, lock in material.items():
        language = languages.get(name)
        data = _locked_source(root, name, lock)
        data = edits.get(name, data)
        if language == 'c' or (language is None and mixed and c_units):
            output = c_root / name
            if language == 'c':
                output = output.with_suffix('.cpp')
                settings[output.as_posix()] = _compile_settings(units[name]['flags'], root, c_root)
            converted, conversions, linkage = _lower_c(name, data, decls[name], casts[name])
            _stage(project, output, converted)
            if language == 'c' or conversions or linkage:
                translation.append(dict(source=name, output=str(output),
                                        c_pointer_conversions=conversions, c_linkage=linkage))
        if language in (None, 'c++'):
            output = cxx_root / name
            _stage(project, output, data)
            if language == 'c++':
                settings[output.as_posix()] = _compile_settings(units[name]['flags'], root, cxx_root)

    if translation:
        (project / 'translation.json').write_text(json.dumps(translation, indent=2) + '\n')
    fortran_libraries = ()
    if fortran:
        from .fortran import emit_fortran

        config = dict(manifest, units=fortran, language='fortran')
        fortran_paths = {unit['path'] for unit in fortran}
        bridge = emit_fortran(config, project, {name: edits[name] for name in edits.keys() & fortran_paths},
                              source_directory='src/fortran-abi', fixed_only=True)
        settings.update({name: ((), bridge['compile_options']) for name in bridge['sources']})
        fortran_libraries = bridge['libraries']

    profiles = (manifest, manifest.get('cxx', {})) if mixed else (
        manifest if c_units else manifest.get('cxx', {}),)
    linkage = {key: tuple(value for profile in profiles for value in profile.get(key, ()))
               for key in ('libraries', 'link_options', 'runtime_dirs')}
    linkage['libraries'] = (*linkage['libraries'], *fortran_libraries)
    common = set(settings.values())
    include_dirs, compile_options = next(iter(common)) if len(common) == 1 else ((), ())
    return dict(sources=tuple(settings), include_dirs=include_dirs, compile_options=compile_options,
                source_settings=settings if len(common) > 1 else {}, **linkage)


def build_reference_project(manifest, project, source_edits=(), *, captures=None):
    """Build the admitted C++20 source revision and check its runtime ABI."""
    edits, previous = {}, {}
    root = Path(manifest['source_root'])
    for edit in reversed(source_edits):
        unit = edit['unit']
        if unit not in edits:
            edits[unit] = _locked_source(root, unit, manifest['source_lock']['files'][unit])
            previous[unit] = len(edits[unit])
        start, stop = edit['span']
        source = edits[unit]
        if (not 0 <= start < stop <= previous[unit]
                or sha256(source[start:stop]).hexdigest() != edit['original_sha256']):
            raise ValueError(f'source copy edit no longer matches {unit}')
        edits[unit] = source[:start] + edit['replacement'].encode('ascii') + source[stop:]
        previous[unit] = start
    project.mkdir()
    build = emit_sources(manifest, project, edits, captures=captures)
    write_native_project(project, 'scar_entry', **build)
    executable, _ = build_native_project(project, 'scar_entry', ('g++',))
    dynamic = cpu_executable_dynamic(executable)
    for name in dynamic['needed']:
        if any(provider in name for provider in ('libpython', 'libtorch', 'libgfortran')):
            raise ValueError(f'unqualified target dependency: {name}')
    return executable, dynamic


def verify_mixed_linkage(manifest, calls, executable):
    """Check that bound cross-language targets have their original link symbols."""
    languages = {unit['path']: unit.get('language', manifest['language']) for unit in manifest['units']}
    if len(set(languages.values())) <= 1:
        return []
    links = read_artifact(calls)['calls']
    targets = sorted({(row['caller']['unit'], row['callee']['unit'], row['linkage_symbol'])
                      for row in links if row['status'] == 'BOUND'
                      and languages[row['caller']['unit']] != languages[row['callee']['unit']]})
    symbols = {line.split()[0] for line in subprocess.check_output(
        ['nm', '-P', '--defined-only', str(executable)], text=True, timeout=30).splitlines() if line.split()}
    for owner, target, symbol in targets:
        if symbol not in symbols:
            raise ValueError(f'mixed linkage symbol {symbol} is absent after building {owner} -> {target}')
    return [dict(caller=owner, callee=target, symbol=symbol) for owner, target, symbol in targets]

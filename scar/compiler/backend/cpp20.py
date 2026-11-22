"""Emit one C++20 project from a locked native source revision."""

from hashlib import sha256
import json
from pathlib import Path

from scar.artifacts import file_reference
from scar.codegen.native_cpu_elf import cpu_executable_dynamic
from scar.codegen.native_project import build_native_project, write_native_project
from scar.ir.execution_recipe import read_artifact
from scar.ir.implementation_plan import PLAN_SCHEMA, SCHEMA as PROBLEM_SCHEMA, problem_document, problem_from_dict
from scar.ir.program.codec import SCHEMA as PROGRAM_SCHEMA
from scar.optimization.joint_planner import verify_execution_plan

from ..frontend.clang import capture_unit
from ..middle.plan import entry_cpus, entry_problem


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
        if role in {'program', 'graph_evidence', 'calls', 'simplified', 'rewrites'}:
            if file_reference(ref['uri'], schema) != ref:
                raise ValueError(f'{role} artifact differs from the plan input')
        else:
            documents[role] = read_artifact(ref)
    observation = documents['cost']
    problem = problem_from_dict(documents['planning_problem'])
    if (observation['schema_version'] != 1 or observation['request'] != artifacts['request']
            or observation['exit'] != 0 or problem_document(entry_problem(entry, artifacts, observation))
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
    manifest = documents['request']
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
                    lowering=option.lowering, verification=verification)
    return manifest, observation, delivery


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
    language = manifest['language']
    units = {unit['path']: unit for unit in manifest['units']}
    if language == 'mixed' or any(unit.get('language', language) != language for unit in units.values()):
        raise ValueError('mixed-language source lowering requires a verified linkage bridge')
    if language == 'fortran':
        from .fortran import emit_fortran
        return emit_fortran(manifest, project, edits)

    root = Path(manifest['source_root']).resolve()
    is_c = language == 'c'
    material = manifest['source_lock']['files']
    if not units or not units.keys() <= material.keys():
        raise ValueError('every translation unit needs locked source bytes')
    if not edits.keys() <= units.keys() or any(not isinstance(value, bytes) for value in edits.values()):
        raise ValueError('source edits must provide byte replacements for translation units')

    settings = {name: _compile_settings(unit['flags'], root) for name, unit in units.items()}
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
    common = set(settings.values())
    if len(common) == 1:
        include_dirs, compile_options = common.pop()
        source_settings = {}
    else:
        include_dirs, compile_options = (), ()
        source_settings = {sources[name]: configuration for name, configuration in settings.items()}
    return dict(sources=tuple(sources[name] for name in units), include_dirs=include_dirs,
                compile_options=compile_options, source_settings=source_settings,
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

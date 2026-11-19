"""Compose checked profile stages into a source-seeded C++20 application."""

from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
from shutil import copy2, copytree
from struct import pack, unpack_from
from tempfile import TemporaryDirectory

from scar.codegen.native_project import build_native_project, write_native_project
from scar.codegen.native_publication_delivery import _local, _read_local, _render
from scar.codegen.native_sparse_source import project_source_csr_stream
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.native_ordered_publication import ENV, FUSED

SCHEMA = 'scar.native-source-triangle-profile'
LAUNCH = 'scar.native-source-triangle-launch'
_EXECUTABLE = 'scar_publish'
_OPTIONS = ('-O2', '-Wall', '-Wextra', '-Werror', '-pedantic')


def _require(valid, reason):
    if not valid:
        raise ValueError(reason)


def _read_member(root, reference):
    name = reference['uri']
    path = root / name
    _require(not Path(name).is_absolute() and path.is_relative_to(root)
        and path.resolve(strict=True) == path, 'profile expected a checked package-relative member')
    return read_artifact({**reference, 'uri': str(path)}, json_document=False)


def _copy_stages(upstream, solver, root):
    """Copy stages without replicating libraries already sharing an inode."""
    copies = {}

    def copy_member(src, dst):
        info = os.stat(src)
        key = (info.st_dev, info.st_ino)
        previous = copies.get(key)
        if previous is None:
            copy2(src, dst)
            copies[key] = dst
        else:
            os.link(previous, dst)
        return dst

    copytree(upstream, root / 'upstream', copy_function=copy_member)
    copytree(solver, root / 'solver', copy_function=copy_member)


def _project(root, reference, profile, source, upstream, solver):
    resource = profile['resources']
    placement = resource['placement']
    count = resource['case']['call_count']
    recipe = resource['recipe_id']
    cpus = placement['launch_cpu_ids']
    owner = placement['owner_cpu_id']
    _require(source['argv'] == [str(count)] and source['iterations'] == count
        and source['target_executed'] is False and owner in cpus
        and cpus == sorted(set(cpus)) and all(type(cpu) is int and 0 <= cpu < 1024 for cpu in cpus),
        'native profile requires one admitted fixed argument, owner and executable CPU mask')
    _require(upstream['recipe_id'] == solver['recipe_id'] == recipe
        and upstream['count'] == solver['count'] == count and upstream['source_count_verified'] is True
        and solver['profile_plan'] == reference and solver['width'] == profile['width']
        and {**upstream['source_seed'], 'uri': profile['source_seed']['uri']} == profile['source_seed'],
        'profile stages do not identify the same executable wire')
    csr = _read_member(root/'upstream', upstream['canonical_csr'])
    stream = project_source_csr_stream(resource, read_artifact(profile['anchor']['detached']))
    header = stream['header']
    expected = pack(header['format'], stream['input']['magic'].encode('ascii'),
        bytes.fromhex(header['recipe_digest_hex']), header['size'], header['nnz'], header['count'], header['flags'])
    fixed = stream['bytes']['input_fixed']
    _require(len(csr) >= fixed and csr.startswith(expected), 'upstream output differs from the solver header')
    rows, = unpack_from('<I', csr, fixed - 4)
    _require(rows <= header['size'] and len(csr) == fixed + stream['bytes']['selected_row'] * rows,
        'prepared CSR rows differ from the solver extent')
    commands = upstream['commands']
    _require(set(commands) == {'topology', 'mesh', 'basis', 'coo'}, 'upstream chain is incomplete')
    prepare = {}
    for stage, role in (('topology','topology'), ('mesh','mesh'), ('basis','basis'), ('ingress','coo')):
        relative, = commands[role]
        path = root / 'upstream' / relative
        _require(not Path(relative).is_absolute() and path.is_relative_to(root/'upstream')
            and path.resolve(strict=True) == path and os.access(path, os.X_OK),
            'profile can launch only checked local stage executables')
        prepare[stage] = dict(command=[f'upstream/{relative}'], cpu_id=owner)
    executable = solver['stream']['executable']['uri']
    _require(executable == 'scar_ordered_sparse' and os.access(root/'solver'/executable, os.X_OK),
        'profile requires the checked canonical solver executable')
    generated = _read_member(root/'upstream', upstream['seed_input'])
    _require(0 < len(generated) <= 4096, 'generated fixed seed does not fit one native input pipe')
    return dict(schema=LAUNCH, schema_version=1, selected=FUSED, commands=[[f'solver/{executable}']],
        cpu_ids=cpus, owner_cpu=owner, pipe_capacity_bytes=0, environment=ENV, cwd='/',
        scope='SOURCE_TRIANGLE_PROFILE_FIXED_ARGV_TO_EXACT_STDOUT', full_python_entry=False,
        source_seed=dict(argv=source['argv'], size_bytes=len(generated), wire_hex=generated.hex(),
            sha256=sha256(generated).hexdigest()), **prepare)


def _providers(root):
    from scar.codegen.native_ordered_profile import check_profile_solver
    from scar.codegen.native_parameterized_upstream import check as check_upstream

    upstream = check_upstream(root/'upstream')
    solver = check_profile_solver(root/'solver')
    return upstream, solver


def _build(root, plan):
    (root/'scar_publish.cpp').write_bytes(_render(plan))
    write_native_project(root, _EXECUTABLE, sources=('scar_publish.cpp',), compile_options=_OPTIONS,
        install_dirs=(('upstream','bin'), ('solver','bin')), preserve_install_permissions=True)
    binary, _ = build_native_project(root, _EXECUTABLE, shlex.split(os.environ.get('CXX', 'c++')))
    binary.rename(root/_EXECUTABLE)


def export(profile_plan, upstream, solver, out):
    from scar.optimization.native_ordered_profile import SCHEMA as PROFILE, verify_profile_plan

    profile_ref = artifact_reference(Path(profile_plan).resolve(strict=True), PROFILE)
    profile = verify_profile_plan(profile_ref['uri'])
    _require(profile == read_artifact(profile_ref), 'profile changed during admission')
    source_ref = profile['source_seed']
    source = read_artifact(source_ref)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-triangle-profile-', dir=out.parent) as temporary:
        root = Path(temporary)/'package'
        root.mkdir()
        _copy_stages(upstream, solver, root)
        checked_up, checked_solver = _providers(root)
        plan = _project(root, profile_ref, profile, source, checked_up, checked_solver)
        (root/'launch.json').write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')
        _build(root, plan)
        document = dict(schema=SCHEMA, schema_version=1, profile_plan=profile_ref,
            source_seed=source_ref, upstream=_local(root, 'upstream/artifact.json', checked_up['schema']),
            solver=_local(root, 'solver/artifact.json', checked_solver['schema']),
            plan=_local(root, 'launch.json', LAUNCH),
            executable=_local(root, _EXECUTABLE, 'scar.native-executable-file'),
            source=_local(root, 'scar_publish.cpp', 'scar.cpp-source'),
            cmake=_local(root, 'CMakeLists.txt', 'scar.cmake-project'),
            fixed_argv=source['argv'], recipe_id=profile['resources']['recipe_id'],
            cpu_ids=plan['cpu_ids'], owner_cpu=plan['owner_cpu'], original_effects_closed=False)
        (root/'artifact.json').write_text(json.dumps(document, indent=2, sort_keys=True) + '\n')
        check(root)
        if out.exists():
            raise FileExistsError(out)
        root.rename(out)
    return artifact_reference(out/'artifact.json', SCHEMA)


def check(package):
    from scar.optimization.native_ordered_profile import verify_profile_plan

    root = Path(package).resolve(strict=True)
    actual = read_artifact(artifact_reference(root/'artifact.json', SCHEMA))
    profile_ref = actual['profile_plan']
    profile = verify_profile_plan(profile_ref['uri'])
    source = read_artifact(actual['source_seed'])
    _require(actual['schema_version'] == 1 and actual['original_effects_closed'] is False
        and read_artifact(profile_ref) == profile and profile['source_seed'] == actual['source_seed'],
        'native launch differs from its admitted source and profile')
    upstream, solver = _providers(root)
    expected = _project(root, profile_ref, profile, source, upstream, solver)
    _require(actual['upstream'] == _local(root, 'upstream/artifact.json', upstream['schema'])
        and actual['solver'] == _local(root, 'solver/artifact.json', solver['schema'])
        and actual['recipe_id'] == profile['resources']['recipe_id']
        and actual['fixed_argv'] == source['argv'] and actual['cpu_ids'] == expected['cpu_ids']
        and actual['owner_cpu'] == expected['owner_cpu']
        and _read_local(root, actual['plan'], 'launch.json', LAUNCH) == expected
        and actual['executable'] == _local(root, _EXECUTABLE, 'scar.native-executable-file')
        and actual['source'] == _local(root, 'scar_publish.cpp', 'scar.cpp-source')
        and actual['cmake'] == _local(root, 'CMakeLists.txt', 'scar.cmake-project')
        and (root/'scar_publish.cpp').read_bytes() == _render(expected)
        and os.access(root/_EXECUTABLE, os.X_OK), 'native package does not reproduce the planned launch')
    return actual

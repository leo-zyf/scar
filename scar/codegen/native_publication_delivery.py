"""Package a replayed ordered publication choice as a standalone C++20 executable."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
from shutil import copytree
from tempfile import TemporaryDirectory

from scar.analysis.replay_transaction import replay_transaction
from scar.codegen import native_coo_ingress
from scar.codegen.native_ordered_sparse_export import SCHEMA as ORDERED_PACKAGE, check_native_ordered_export
from scar.codegen.native_project import build_native_project, write_native_project
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.native_ordered_publication import BASE, FUSED, SCHEMA as CHOICE, check as check_choice
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.native-ordered-publication-delivery'
PLAN = 'scar.native-ordered-publication-launch-plan'
_TEMPLATE = Path(__file__).with_name('native_publication_supervisor_v1.cpp.in')
_TARGET = 'scar_publish'
_ARMS = ('binary', 'digest')
_INPUT_STAGES = {
    'topology': ('topology', 'bin/scar_triangle_topology', 'mesh_data', 'triangle seed'),
    'mesh': ('mesh', 'bin/scar_mesh_basis', 'basis_data', 'mesh arrays'),
    'basis': ('basis', 'bin/scar_basis_coo', 'operands', 'basis arrays'),
    'ingress': ('coo', 'bin/scar_coo_ingress', 'canonical', 'COO operands'),
}
_FLAGS = ('-O2', '-Wall', '-Wextra', '-Werror', '-pedantic')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _local(root, name, schema):
    path = root / name
    _require(path.resolve(strict=True) == path, 'expected a regular local publication member: ' + name)
    return {**artifact_reference(path, schema), 'uri': name}


def _same_content(ref, expected):
    return {**ref, 'uri': expected['uri']} == expected


def _read_local(root, actual, name, schema):
    _require(actual == _local(root, name, schema), 'local publication evidence differs from its receipt')
    return read_artifact({**actual, 'uri': str(root / name)})


def _project(choice, design, packages, stages):
    launch, options = design['launch'], design['graph']['options']
    cpus, owner = launch['cpu_ids'], launch['consumer_cpu_id']
    capacity = options[BASE]['pipe_capacity_bytes']
    selected = choice['selected']
    _require(choice['observations'] is not None and choice['timing']['same_exact_stdout'] is True
        and choice['timing']['same_no_stderr'] is True and choice['timing']['measured_output_bytes'] == 65
        and choice['deployment_authorized'] is False and selected in (BASE, FUSED)
        and launch['consumer_uses_additional_core'] is False and owner in cpus
        and cpus == sorted(set(cpus)) and all(type(cpu) is int and 0 <= cpu < 1024 for cpu in cpus)
        and type(capacity) is int and 0 < capacity < 2**31,
        'delivery requires a measured independent stdout choice and its physical CPU/pipe configuration')
    environment = launch['environment']
    _require(type(environment) is dict and all(re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', key)
        and type(value) is str and value.isascii() and '\0' not in value for key, value in environment.items())
        and type(launch['cwd']) is str and Path(launch['cwd']).is_absolute() and launch['cwd'].isascii()
        and '\0' not in launch['cwd'], 'invalid frozen publication environment or current directory')
    _require(len(design['packages']) == len(packages) == 2, 'the ordered publication choice must bind two arms')
    commands = []
    for role, arm, package in zip((BASE, FUSED), _ARMS, packages, strict=True):
        target = package['stream']['executable']
        _require(_same_content(target, design['executables'][role])
            and target['uri'] == 'scar_ordered_sparse'
            and package['qualified_release']['sha256'] == design['release_sha256'],
            'the bundled executable or release differs from the measured publication')
        commands.append(f'{arm}/{target["uri"]}')
    _require([package['schema_version'] for package in packages] == [4, 5],
        'delivery requires canonical binary and digest packages')
    plan = dict(schema=PLAN, schema_version=1, selected=selected, decision=choice['decision'],
        source=design['input'], release_sha256=design['release_sha256'], publication=design['publication'],
        commands=[[commands[0]], [commands[1], '--consume-records']] if selected == BASE else [[commands[1]]],
        cpu_ids=cpus, owner_cpu=owner, pipe_capacity_bytes=capacity if selected == BASE else 0,
        environment=environment, cwd=launch['cwd'], timeout_ns=120_000_000_000,
        completion='BUFFER_ONE_DIGEST_LINE_AND_REQUIRE_EVERY_CHILD_EXIT_ZERO_BEFORE_STDOUT',
        scope='EXPLICIT_INDEPENDENT_CANONICAL_CSR_TO_SHA256_STDOUT', full_python_entry=False)
    if 'ingress' in stages:
        reference, coo = stages['ingress']
        _require(_same_content(coo['canonical_sample'], plan['source']),
            'COO observation does not produce the canonical CSR input of this publication choice')
        plan.update(schema_version=2, scope='EXPLICIT_COO_TO_SHA256_STDOUT',
            ingress=dict(receipt=reference, command=['coo/bin/scar_coo_ingress'], cpu_id=owner,
                contract='COO_ASSEMBLY_TO_CANONICAL_CSR_ON_STDIN'))
    if 'basis' in stages:
        reference, upstream = stages['basis']
        _require('ingress' in stages and coo['input_sample'] == upstream['original_coo']
            and upstream['executable']['uri'] == 'bin/scar_basis_coo',
            'basis provider output must be the exact input of the selected COO stage')
        plan.update(schema_version=3, scope='EXPLICIT_BASIS_ARRAYS_TO_SHA256_STDOUT',
            basis=dict(receipt=reference, command=['basis/bin/scar_basis_coo'], cpu_id=owner,
                source=upstream['admitted_input'],
                contract='BASIS_ARRAYS_TO_COO_ON_STDIN'))
    for stage, required, result, version, scope, contract in (
            ('mesh', 'basis', 'exact_basis', 4, 'EXPLICIT_MESH_ARRAYS_TO_SHA256_STDOUT',
                'MESH_ARRAYS_TO_BASIS_ON_STDIN'),
            ('topology', 'mesh', 'exact_mesh', 5, 'EXPLICIT_TRIANGLE_SEED_TO_SHA256_STDOUT',
                'TRIANGLE_SEED_TO_MESH_ARRAYS_ON_STDIN')):
        if stage not in stages:
            continue
        reference, upstream = stages[stage]
        directory, target, _, _ = _INPUT_STAGES[stage]
        _require(required in stages and upstream[result] == plan[required]['source']
            and upstream['executable']['uri'] == target,
            'upstream stage output must be the exact input of its admitted consumer: ' + stage)
        plan.update(schema_version=version, scope=scope)
        plan[stage] = dict(receipt=reference, command=[f'{directory}/{target}'], cpu_id=owner,
            source=upstream['admitted_input'], contract=contract)
    return plan


def _render(plan):
    split = plan['selected'] == BASE
    producer = plan['commands'][0][0] if split else ''
    digest = plan['commands'][-1][0]
    environment = ''.join(f'const_cast<char*>({json.dumps(key + "=" + value)}), '
        for key, value in sorted(plan['environment'].items()))
    values = dict(SPLIT='true' if split else 'false', CPUS=', '.join(map(str, plan['cpu_ids'])),
        OWNER=str(plan['owner_cpu']), PIPE=str(plan['pipe_capacity_bytes']), ENV=environment,
        CWD=json.dumps(plan['cwd']), PRODUCER=json.dumps(producer), DIGEST=json.dumps(digest),
        START_INGRESS='', CLOSE_INGRESS='', WAIT_INGRESS='', INPUT='STDIN_FILENO',
        INLET='canonical CSR')
    sequence = [stage for stage in _INPUT_STAGES if stage in plan]
    if sequence:
        pin_owner = '    set_affinity(std::span(&owner, 1));\n'
        start = [pin_owner] if len(sequence) > 1 else []
        previous = None
        for stage in sequence:
            _, _, channel, _ = _INPUT_STAGES[stage]
            command, = plan[stage]['command']
            input_fd = 'STDIN_FILENO' if previous is None else f'{previous}[0].get()'
            start.extend((f'    Child {stage};\n', f'    auto {channel} = make_pipe();\n'))
            if len(sequence) == 1:
                start.append(pin_owner)
            start.extend((f'    {stage}.launch(root / {json.dumps(command)}, nullptr, {input_fd}, {channel}[1].get());\n',
                          f'    {channel}[1].close_now();\n'))
            if previous is not None:
                start.append(f'    {previous}[0].close_now();\n')
            previous = channel
        values.update(START_INGRESS=''.join(start), CLOSE_INGRESS=f'    {previous}[0].close_now();\n',
            WAIT_INGRESS=''.join(f'    {stage}.successful(end);\n' for stage in reversed(sequence)),
            INPUT=f'{previous}[0].get()', INLET=_INPUT_STAGES[sequence[0]][3])
    source = _TEMPLATE.read_text()
    for marker, value in values.items():
        source = source.replace(f'@SCAR_{marker}@', value)
    return source.encode()


def _copy_evidence(directory, label, reference):
    name = f'evidence/{label}.json'
    (directory / name).write_bytes(read_artifact(reference, json_document=False))
    return _local(directory, name, reference['schema'])


def _build(directory, plan):
    (directory / 'scar_publish.cpp').write_bytes(_render(plan))
    directories = (*_ARMS, *(_INPUT_STAGES[stage][0] for stage in reversed(_INPUT_STAGES) if stage in plan))
    write_native_project(directory, _TARGET, sources=('scar_publish.cpp',), compile_options=_FLAGS,
        install_dirs=tuple((path, 'bin') for path in directories), preserve_install_permissions=True)
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    executable, _ = build_native_project(directory, _TARGET, compiler)
    executable.rename(directory / _TARGET)


def _input_member(root, stage):
    if stage == 'ingress':
        module = native_coo_ingress
    elif stage == 'basis':
        from scar.codegen import native_basis_coo as module
    elif stage == 'mesh':
        from scar.codegen import native_mesh_basis as module
    else:
        from scar.codegen import native_triangle_topology as module

    directory = _INPUT_STAGES[stage][0]
    return _local(root, f'{directory}/artifact.json', module.SCHEMA), module.check(root / directory)


def _link_identical_providers(root, stages):
    shared = {}
    for stage, (_, document) in stages.items():
        folder = root / _INPUT_STAGES[stage][0]
        for reference in document.get('libraries', ()):
            identity = reference['sha256'], reference['size_bytes']
            current = folder / reference['uri']
            previous = shared.setdefault(identity, current)
            if current != previous:
                current.unlink()
                current.hardlink_to(previous)


@replay_transaction()
def export(choice, out, *, coo_observation=None, scipy_source=None, basis_package=None,
           mesh_package=None, topology_package=None):
    _require((coo_observation is None) == (scipy_source is None), 'COO delivery requires both provider inputs')
    _require(basis_package is None or coo_observation is not None, 'basis delivery also requires the COO stage')
    _require(mesh_package is None or basis_package is not None, 'mesh delivery also requires the Basis stage')
    _require(topology_package is None or mesh_package is not None, 'topology delivery also requires the mesh stage')
    choice = Path(choice).resolve(strict=True)
    reference = artifact_reference(choice, CHOICE)
    selected = check_choice(choice)
    read_artifact(reference, json_document=False)
    design_ref = selected['design']
    design = read_artifact(design_ref)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-publication-delivery-', dir=out.parent) as temporary:
        root = Path(temporary)
        (root / 'evidence').mkdir()
        evidence = dict(choice=_copy_evidence(root, 'choice', reference),
            design=_copy_evidence(root, 'design', design_ref))
        observations = selected['observations']
        _require(observations is not None, 'an unmeasured publication choice cannot be delivered')
        evidence['observations'] = _copy_evidence(root, 'observations', observations)
        bundles, packages = [], []
        for role, member in zip(_ARMS, design['packages'], strict=True):
            source = Path(member['uri']).parent
            copytree(source, root / role)
            reference = _local(root, f'{role}/artifact.json', ORDERED_PACKAGE)
            _require(_same_content(reference, member), 'copied arm differs from the original publication choice')
            bundles.append(reference)
            packages.append(check_native_ordered_export(root / role))
        stages = {}
        if coo_observation is not None:
            native_coo_ingress.export(coo_observation, scipy_source, root / 'coo')
            stages['ingress'] = _input_member(root, 'ingress')
        for stage, package in (('basis', basis_package), ('mesh', mesh_package), ('topology', topology_package)):
            if package is None:
                continue
            copytree(Path(package).resolve(strict=True), root / _INPUT_STAGES[stage][0])
            stages[stage] = _input_member(root, stage)
        plan = _project(selected, design, packages, stages)
        _link_identical_providers(root, stages)
        (root / 'plan.json').write_text(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + '\n')
        _build(root, plan)
        document = dict(schema=SCHEMA, schema_version=plan['schema_version'], evidence=evidence, packages=bundles,
            selected=plan['selected'], plan=_local(root, 'plan.json', PLAN),
            source=_local(root, 'scar_publish.cpp', 'scar.cpp-source'),
            executable=_local(root, _TARGET, 'scar.native-executable-file'),
            cmake=_local(root, 'CMakeLists.txt', 'scar.cmake-project'), full_python_entry=False)
        for stage, (reference, _) in stages.items():
            document[stage] = reference
        (root / 'artifact.json').write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + '\n')
        check(root)
        if out.exists():
            raise FileExistsError(out)
        root.rename(out)
    return artifact_reference(out / 'artifact.json', SCHEMA)


def check(package):
    root = Path(package).resolve(strict=True)
    doc = _read_local(root, _local(root, 'artifact.json', SCHEMA), 'artifact.json', SCHEMA)
    ordered = tuple(reversed(_INPUT_STAGES))
    version = doc['schema_version']
    _require(type(version) is int and 1 <= version <= 5
        and tuple(stage for stage in ordered if stage in doc) == ordered[:version - 1]
        and doc['full_python_entry'] is False, 'unsupported native publication delivery')
    evidence = doc['evidence']
    selected = _read_local(root, evidence['choice'], 'evidence/choice.json', CHOICE)
    design_ref, observations = selected['design'], selected['observations']
    _require(_same_content(evidence['design'], design_ref) and _same_content(evidence['observations'], observations),
        'copied publication decision dependencies differ')
    design = _read_local(root, evidence['design'], 'evidence/design.json', design_ref['schema'])
    _read_local(root, evidence['observations'], 'evidence/observations.json', observations['schema'])
    packages = []
    for role, archived, expected in zip(_ARMS, doc['packages'], design['packages'], strict=True):
        _require(archived == _local(root, f'{role}/artifact.json', ORDERED_PACKAGE) and _same_content(archived, expected),
            'local ordered publication package differs from its measured arm')
        packages.append(check_native_ordered_export(root / role))
    plan = _read_local(root, doc['plan'], 'plan.json', PLAN)
    stages = {}
    for stage in ordered[:version - 1]:
        stages[stage] = _input_member(root, stage)
        _require(doc[stage] == stages[stage][0], 'bundled provider differs from publication receipt: ' + stage)
    _require(doc['selected'] == plan['selected']
        and semantic_sha256(plan) == semantic_sha256(_project(selected, design, packages, stages))
        and doc['source'] == _local(root, 'scar_publish.cpp', 'scar.cpp-source')
        and (root / 'scar_publish.cpp').read_bytes() == _render(plan)
        and doc['cmake'] == _local(root, 'CMakeLists.txt', 'scar.cmake-project')
        and doc['executable'] == _local(root, _TARGET, 'scar.native-executable-file')
        and os.access(root / _TARGET, os.X_OK), 'generated publication project differs from its replayed plan')
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--choice', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--coo-observation', type=Path)
    parser.add_argument('--scipy-source', type=Path)
    parser.add_argument('--basis-package', type=Path)
    parser.add_argument('--mesh-package', type=Path)
    parser.add_argument('--topology-package', type=Path)
    args = parser.parse_args(argv)
    if args.check and all(value is None for value in (args.out, args.coo_observation,
            args.scipy_source, args.basis_package, args.mesh_package, args.topology_package)):
        receipt = check(args.check)
        print(json.dumps(dict(verified=True, selected=receipt['selected'])))
    elif args.choice and args.out:
        print(json.dumps(export(args.choice, args.out, coo_observation=args.coo_observation,
            scipy_source=args.scipy_source, basis_package=args.basis_package,
            mesh_package=args.mesh_package, topology_package=args.topology_package)))
    else:
        parser.error('choose --check PACKAGE or --choice RESULT --out PACKAGE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

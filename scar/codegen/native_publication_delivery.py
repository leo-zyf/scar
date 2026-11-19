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
_INGRESS = 'coo/bin/scar_coo_ingress'
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


def _project(choice, design, packages, ingress=None):
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
    if ingress is not None:
        reference, coo = ingress
        _require(_same_content(coo['canonical_sample'], plan['source']),
            'COO observation does not produce the canonical CSR input of this publication choice')
        plan.update(schema_version=2, scope='EXPLICIT_COO_TO_SHA256_STDOUT',
            ingress=dict(receipt=reference, command=[_INGRESS], cpu_id=owner,
                contract='COO_ASSEMBLY_TO_CANONICAL_CSR_ON_STDIN'))
    return plan


def _render(plan):
    split = plan['selected'] == BASE
    producer = plan['commands'][0][0] if split else ''
    digest = plan['commands'][-1][0]
    environment = ''.join(f'const_cast<char*>({json.dumps(key + "=" + value)}), '
        for key, value in sorted(plan['environment'].items()))
    coo = plan.get('ingress')
    values = dict(SPLIT='true' if split else 'false', CPUS=', '.join(map(str, plan['cpu_ids'])),
        OWNER=str(plan['owner_cpu']), PIPE=str(plan['pipe_capacity_bytes']), ENV=environment,
        CWD=json.dumps(plan['cwd']), PRODUCER=json.dumps(producer), DIGEST=json.dumps(digest),
        START_INGRESS='', CLOSE_INGRESS='', WAIT_INGRESS='', INPUT='STDIN_FILENO',
        INLET='canonical CSR')
    if coo is not None:
        command, = coo['command']
        values.update(START_INGRESS=(
            '    Child ingress;\n'
            '    auto canonical = make_pipe();\n'
            '    set_affinity(std::span(&owner, 1));\n'
            f'    ingress.launch(root / {json.dumps(command)}, nullptr, STDIN_FILENO, canonical[1].get());\n'
            '    canonical[1].close_now();\n'),
            CLOSE_INGRESS='    canonical[0].close_now();\n', WAIT_INGRESS='    ingress.successful(end);\n',
            INPUT='canonical[0].get()', INLET='COO operands')
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
    directories = (*_ARMS, 'coo') if 'ingress' in plan else _ARMS
    write_native_project(directory, _TARGET, sources=('scar_publish.cpp',), compile_options=_FLAGS,
        install_dirs=tuple((path, 'bin') for path in directories), preserve_install_permissions=True)
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    executable, _ = build_native_project(directory, _TARGET, compiler)
    executable.rename(directory / _TARGET)


@replay_transaction()
def export(choice, out, *, coo_observation=None, scipy_source=None):
    _require((coo_observation is None) == (scipy_source is None), 'COO delivery requires both provider inputs')
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
        ingress = None
        if coo_observation is not None:
            native_coo_ingress.export(coo_observation, scipy_source, root / 'coo')
            ingress = (_local(root, 'coo/artifact.json', native_coo_ingress.SCHEMA),
                       native_coo_ingress.check(root / 'coo'))
        plan = _project(selected, design, packages, ingress)
        (root / 'plan.json').write_text(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + '\n')
        _build(root, plan)
        document = dict(schema=SCHEMA, schema_version=1, evidence=evidence, packages=bundles,
            selected=plan['selected'], plan=_local(root, 'plan.json', PLAN),
            source=_local(root, 'scar_publish.cpp', 'scar.cpp-source'),
            executable=_local(root, _TARGET, 'scar.native-executable-file'),
            cmake=_local(root, 'CMakeLists.txt', 'scar.cmake-project'), full_python_entry=False)
        if ingress is not None:
            document.update(schema_version=2, ingress=ingress[0])
        (root / 'artifact.json').write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + '\n')
        check(root)
        if out.exists():
            raise FileExistsError(out)
        root.rename(out)
    return artifact_reference(out / 'artifact.json', SCHEMA)


def check(package):
    root = Path(package).resolve(strict=True)
    doc = _read_local(root, _local(root, 'artifact.json', SCHEMA), 'artifact.json', SCHEMA)
    _require((doc['schema_version'], 'ingress' in doc) in ((1, False), (2, True))
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
    ingress = None
    if 'ingress' in doc:
        ref = _local(root, 'coo/artifact.json', native_coo_ingress.SCHEMA)
        _require(doc['ingress'] == ref, 'bundled COO provider differs from publication receipt')
        ingress = ref, native_coo_ingress.check(root / 'coo')
    _require(doc['selected'] == plan['selected']
        and semantic_sha256(plan) == semantic_sha256(_project(selected, design, packages, ingress))
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
    args = parser.parse_args(argv)
    if args.check and args.out is None and args.coo_observation is None and args.scipy_source is None:
        receipt = check(args.check)
        print(json.dumps(dict(verified=True, selected=receipt['selected'])))
    elif args.choice and args.out:
        print(json.dumps(export(args.choice, args.out, coo_observation=args.coo_observation,
            scipy_source=args.scipy_source)))
    else:
        parser.error('choose --check PACKAGE or --choice RESULT --out PACKAGE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

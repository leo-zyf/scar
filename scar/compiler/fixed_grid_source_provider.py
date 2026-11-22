"""Qualify a source-bound fixed-grid recurrence and compile a detached CPU entry."""

import argparse
import json
from math import prod
from pathlib import Path
import subprocess
from time import perf_counter_ns

from scar.codegen.native_cpu_package import export_native_cpu, verify_native_cpu_package
from scar.compiler.compile_aten_recurrence import compile_aten_recurrence
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.validate_native_project import verify_native_project


SCHEMA = 'scar.cpu-fixed-grid-source-provider'
RESULT_SCHEMA = 'scar.cpu-fixed-grid-entry-compilation'
PROJECT_SCHEMA = 'scar.aten-recurrence-delivery'
RUNTIME_SCHEMA = 'scar.native-cpu-package'
SOURCE_SCHEMA = 'scar.python-source'
EXECUTABLE_SCHEMA = 'application/x-executable'
OUTPUT_SCHEMA = 'scar.native-float32-output'
_SCOPE = 'FIXED_GRID_FINAL_TENSOR_FILE_NORMAL_COMPLETION'
_ENV = {'PATH': '/usr/bin:/bin', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def _source_project(reference):
    if reference['schema'] != PROJECT_SCHEMA or Path(reference['uri']).name != 'artifact.json':
        raise ValueError('a recurrence project manifest is required')
    manifest = read_artifact(reference)
    directory, proof = verify_native_project(Path(reference['uri']).parent, reference['sha256'])
    if manifest['cxx_standard'] != 20 or not manifest['source_entry_replayable'] or 'cpu_linkage' not in proof:
        raise ValueError('an admitted, source-bound C++20 CPU recurrence is required')
    source = json.loads((directory / 'plan/source-entry.json').read_bytes())
    name = source['source']
    entry = _ref(name['path'], SOURCE_SCHEMA)
    if (entry['sha256'], entry['size_bytes']) != (name['sha256'], name['size_bytes']):
        raise ValueError('fixed-grid source differs from the verified project')
    return manifest, source, entry


def _input_refs(selection):
    rows = read_artifact(selection)['inputs']
    refs = [_ref(row['path'], OUTPUT_SCHEMA) for row in rows]
    if any((ref['uri'], ref['sha256'], ref['size_bytes']) !=
           (row['path'], row['sha256'], row['size_bytes']) for row, ref in zip(rows, refs, strict=True)):
        raise ValueError('fixed-grid operands differ from the admitted native order and bytes')
    return refs


def _directory(path):
    value = Path(path).resolve(strict=True)
    if not value.is_dir():
        raise ValueError('an existing directory is required: ' + str(value))
    return str(value)


def _authority(project, interpreter, cwd, pythonpath, calibration_steps):
    if type(calibration_steps) is not int or not 1 <= calibration_steps <= 64:
        raise ValueError('fixed-grid calibration steps must be within [1, 64]')
    manifest, source, entry = _source_project(project)
    selection = manifest['source_references']['selection']
    inputs = _input_refs(selection)
    directories = {str(Path(row['uri']).parent) for row in inputs}
    if len(directories) != 1:
        raise ValueError('the source CLI contract requires file inputs from one directory')
    input_directory, = directories
    # Retain the invocation path. Resolving a venv interpreter before exec
    # would select a different sys.prefix despite identical interpreter bytes.
    interpreter = artifact_reference(Path(interpreter).absolute(), EXECUTABLE_SCHEMA)
    paths = [_directory(path) for path in pythonpath]
    if any(':' in path for path in paths):
        raise ValueError('source replay PYTHONPATH entries cannot contain a separator')
    return dict(schema=SCHEMA, schema_version=1, entry=entry, source_project=project,
                source_binding=source, context=dict(interpreter=interpreter, cwd=_directory(cwd), pythonpath=paths),
                inputs=inputs, input_directory=input_directory, source_cli=['INPUT_DIRECTORY', 'STEP_COUNT', 'OUTPUT_PATH'],
                selection=selection, geometry_contract=manifest['source_references']['geometry_contract'],
                tensor_budget_bytes=manifest['tensor_budget_bytes'], calibration_steps=calibration_steps,
                scope=_SCOPE)


def register(project, *, interpreter, cwd, pythonpath, calibration_steps, out):
    value = _authority(_ref(project, PROJECT_SCHEMA), interpreter, cwd, pythonpath, calibration_steps)
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    _write(out, value)
    return _ref(out, SCHEMA)


def check(path):
    value = read_artifact(_ref(path, SCHEMA))
    context = value['context']
    if value != _authority(value['source_project'], context['interpreter']['uri'], context['cwd'],
                           context['pythonpath'], value['calibration_steps']):
        raise ValueError('fixed-grid provider differs from its qualified source and recurrence')
    return value


def _argv(authority, argv, *, require_fresh=False):
    argv = list(argv)
    if len(argv) != 3 or any(type(item) is not str or not item or '\0' in item for item in argv):
        raise ValueError('fixed-grid source entry takes INPUT_DIRECTORY STEP_COUNT OUTPUT_PATH')
    root, digits, output = argv
    if (_directory(Path(authority['context']['cwd']) / root) != authority['input_directory']
            or len(digits) > 10 or not digits.isascii() or not digits.isdecimal()
            or not 1 <= int(digits) <= 2**31 - 1 or digits != str(int(digits))):
        raise ValueError('source profile requires admitted inputs and a positive native step count')
    requested = Path(authority['context']['cwd']) / output
    if require_fresh and (requested.is_symlink() or requested.exists()):
        raise ValueError('the fixed-grid output contract requires a fresh destination')
    return argv


def _native_call(runtime, package, project, authority, argv):
    return dict(executable=package['executable'], package=runtime, cpus=project['cpus'],
                arguments=[argv[1], *(row['uri'] for row in authority['inputs']), argv[2]],
                runtime='LINUX_CXX20_CPU_LIBTORCH', environment=dict(_ENV))


def _launch(command, cpus, cwd, environment, directory, stem):
    prefix = ['/usr/bin/taskset', '-c', ','.join(map(str, cpus))]
    started = perf_counter_ns()
    actual = subprocess.run([*prefix, *command], cwd=cwd, env=environment, capture_output=True, timeout=900)
    elapsed = perf_counter_ns() - started
    if actual.returncode:
        raise ValueError(f'{stem} failed: ' + actual.stderr.decode(errors='replace').strip())
    roles = [('stdout', actual.stdout), ('stderr', actual.stderr)]
    result = dict(wall_ns_single_observation=elapsed, returncode=actual.returncode,
                  output=_ref(directory / (stem + '.f32'), OUTPUT_SCHEMA))
    for role, raw in roles:
        target = directory / (stem + '.' + role)
        with target.open('xb') as stream:
            stream.write(raw)
        result[role] = _ref(target, 'application/octet-stream')
    return result


def _byte_match(left, right):
    return all(left[key] == right[key] for key in ('sha256', 'size_bytes'))


def _validation(directory, project, original, native):
    execution = json.loads((Path(project['uri']).parent / 'plan/execution.json').read_bytes())
    output, = [row for row in execution['operations'] if row['id'] == execution['output']]
    nbytes = 4 * prod(output['shape'])
    metadata = dict(abi_version=1, dtype='float32', byte_order='little',
                    outputs=[dict(shape=output['shape'], nbytes=nbytes)])
    if (output['dtype'] != 'FLOAT32' or native['output']['size_bytes'] != nbytes
            or not _byte_match(original['output'], native['output'])
            or json.loads(read_artifact(native['stdout'], json_document=False)) != metadata):
        raise ValueError('fresh source and native fixed-grid final Tensor verification failed')
    for stem, run in (('original', original), ('native', native)):
        if (set(run) != {'returncode', 'wall_ns_single_observation', 'output', 'stdout', 'stderr'}
                or type(run['returncode']) is not int or run['returncode'] != 0 or run['stderr']['size_bytes']
                or type(run['wall_ns_single_observation']) is not int or run['wall_ns_single_observation'] <= 0):
            raise ValueError('fixed-grid normal completion evidence differs from its scope')
        for key, suffix, schema in (('output', 'f32', OUTPUT_SCHEMA), ('stdout', 'stdout', 'application/octet-stream'),
                                    ('stderr', 'stderr', 'application/octet-stream')):
            if run[key] != _ref(directory / f'{stem}.{suffix}', schema):
                raise ValueError('fixed-grid verification files differ from their exact local references')
    return dict(original=original, native=native, bitwise_final_tensor=True,
                stdout_contract='SOURCE_STATUS_AND_NATIVE_TENSOR_ABI_METADATA_CHECKED_SEPARATELY',
                timing='ONE_CLEAN_RUN_PER_SIDE_NO_SPEEDUP_VERDICT')


def compile_entry(provider_path, entry, argv, out):
    provider = _ref(provider_path, SCHEMA)
    authority = check(provider['uri'])
    source = _ref(entry, SOURCE_SCHEMA)
    if source != authority['entry']:
        raise ValueError('entry does not match the source bound by the fixed-grid provider')
    argv = _argv(authority, argv, require_fresh=True)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    context = authority['context']
    source_environment = dict(_ENV, PYTHONPATH=':'.join(context['pythonpath']))
    discovery = authority['source_binding']['discovery']
    selection, geometry = authority['selection'], authority['geometry_contract']
    tuning = [context['interpreter']['uri'], '-B', '-m', 'scar', 'compile-aten-recurrence',
              '--selection', selection['uri'], '--selection-sha256', selection['sha256'],
              '--geometry-contract', geometry['uri'], '--geometry-contract-sha256', geometry['sha256'],
              '--discovery', discovery['uri'], '--discovery-sha256', discovery['sha256'],
              '--source-entry', authority['source_binding']['entry'], '--calibration-steps',
              str(authority['calibration_steps']), '--tensor-budget-bytes', str(authority['tensor_budget_bytes']),
              '--require-cpu-only', '--out', str(out / 'project')]
    # A fresh interpreter is needed to bind torch and the ODE provider to the
    # admitted PYTHONPATH before any optional Python backend imports.
    read_artifact(context['interpreter'], json_document=False)
    build_env = dict(source_environment, PYTHONPATH=':'.join([str(Path(__file__).resolve().parents[2]),
                                                             *context['pythonpath']]))
    build = subprocess.run(tuning, cwd=context['cwd'], env=build_env, capture_output=True, timeout=900)
    (out / 'build.stdout').write_bytes(build.stdout)
    (out / 'build.stderr').write_bytes(build.stderr)
    if build.returncode:
        raise ValueError('fresh fixed-grid recurrence build failed: ' + build.stderr.decode(errors='replace').strip())
    project = _ref(out / 'project/artifact.json', PROJECT_SCHEMA)
    compiled = read_artifact(project)
    export_native_cpu(out / 'project', project['sha256'], out=out / 'runtime stage')
    # Exercise ELF lookup after moving the package to the delivered location.
    runtime_dir = (out / 'runtime stage').rename(out / 'CPU runtime')
    runtime = _ref(runtime_dir / 'artifact.json', RUNTIME_SCHEMA)
    verify_native_cpu_package(runtime_dir, runtime['sha256'])
    package = read_artifact(runtime)
    invocation = _native_call(runtime, package, compiled, authority, argv)
    directory = out / 'verification'
    directory.mkdir()
    original = _launch([context['interpreter']['uri'], '-B', source['uri'], *argv[:2],
                        str(directory / 'original.f32')], invocation['cpus'], context['cwd'],
                       source_environment, directory, 'original')
    native = _launch([str(runtime_dir / invocation['executable']), *invocation['arguments'][:-1],
                      str(directory / 'native.f32')], invocation['cpus'], context['cwd'], _ENV, directory, 'native')
    validation = _validation(directory, project, original, native)
    _input_refs(selection)
    read_artifact(source, json_document=False)
    result = dict(schema=RESULT_SCHEMA, schema_version=1, provider=provider, entry=source, argv=argv,
                  scope=_SCOPE, project=project, runtime=runtime, invocation=invocation,
                  validation=validation)
    record = out / 'result.json'
    _write(record, result)
    return _ref(record, RESULT_SCHEMA)


def check_entry(path):
    reference = _ref(path, RESULT_SCHEMA)
    result = read_artifact(reference)
    folder = Path(reference['uri']).parent
    provider = result['provider']
    authority = check(provider['uri'])
    read_artifact(provider)
    if (result['schema_version'] != 1 or result['scope'] != _SCOPE or provider['schema'] != SCHEMA
            or result['entry'] != authority['entry']):
        raise ValueError('fixed-grid entry receipt differs from its authority')
    _argv(authority, result['argv'])
    if (result['project'] != _ref(folder / 'project/artifact.json', PROJECT_SCHEMA)
            or result['runtime'] != _ref(folder / 'CPU runtime/artifact.json', RUNTIME_SCHEMA)):
        raise ValueError('fixed-grid receipt must refer to its own local project and runtime')
    project, source, entry = _source_project(result['project'])
    runtime = result['runtime']
    package = read_artifact(runtime)
    verify_native_cpu_package(Path(runtime['uri']).parent, runtime['sha256'])
    invocation = _native_call(runtime, package, project, authority, result['argv'])
    if (source != authority['source_binding'] or entry != result['entry']
            or project['source_references']['selection'] != authority['selection']
            or project['source_references']['geometry_contract'] != authority['geometry_contract']
            or project['tensor_budget_bytes'] != authority['tensor_budget_bytes']
            or project['calibration_steps'] != authority['calibration_steps']
            or package['source_project']['manifest_sha256'] != result['project']['sha256']
            or result['invocation'] != invocation):
        raise ValueError('fixed-grid project and runtime differ from the admitted lowering')
    validation = result['validation']
    original, native = validation['original'], validation['native']
    if validation != _validation(folder / 'verification', result['project'], original, native):
        raise ValueError('fixed-grid validation differs from its supported evidence contract')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    record = commands.add_parser('register')
    record.add_argument('--project', required=True)
    record.add_argument('--interpreter', required=True)
    record.add_argument('--cwd', required=True)
    record.add_argument('--pythonpath', action='append', default=[])
    record.add_argument('--calibration-steps', type=int, default=2)
    record.add_argument('--out', required=True)
    for mode in ('check', 'check-entry'):
        command = commands.add_parser(mode)
        command.add_argument('path')
    args = vars(parser.parse_args(argv))
    mode = args.pop('mode')
    if mode == 'register':
        result = register(**args)
    else:
        result = (check if mode == 'check' else check_entry)(**args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()

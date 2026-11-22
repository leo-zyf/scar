"""Qualify source and argv for a retained native array task entry."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from scar.codegen import retain_native_array_task_entry as retention
from scar.compiler.calibrate_array_entry import _ref, _write
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.cpu-array-task-source-provider'
RESULT_SCHEMA = 'scar.cpu-array-task-entry-compilation'
_SOURCE = 'scar.python-source'
_RECEIPT = 'result.json'


def _require(condition, detail):
    if not condition:
        raise ValueError('native array task entry: ' + detail)


def _profile(reference):
    _require(reference['schema'] == retention.SCHEMA, 'expected a retained native entry')
    kept = read_artifact(reference)
    replay = retention.resolve(kept['schedule']['uri'], kept['existing_acceptance']['uri'], archived=kept)
    _require(kept == replay, 'retained entry changed during cold qualification')
    accepted = read_artifact(kept['existing_acceptance'])
    runtime = kept['retained_release']
    package = read_artifact(runtime)
    recipe_ref = package['recipe']
    recipe = read_artifact(recipe_ref)
    source = recipe['entry']
    holdout_ref = accepted['holdout']
    holdout = read_artifact(holdout_ref)
    argv = holdout['argv']
    flag, = recipe['options'][recipe['output_option']]['flags']
    leaf = argv[-1]
    cwd = Path(holdout['stdout']['uri']).parent
    original = holdout['original_run']
    executable = package['files'][package['binary']]
    interpreter = recipe['source_python']
    cpus = package['cpu_ids']
    environment = holdout['environment']

    _require(source == _ref(source['uri'], _SOURCE) == holdout['entry']
             and holdout['recipe'] == recipe_ref and holdout['interpreter'] == interpreter
             and cpus == holdout['cpu_ids'] and environment == recipe['environment']
             and holdout['valid_for'] == 'VALIDATION_ONLY'
             and holdout['independent_original_run'] is True and holdout['instrumented'] is False
             and holdout['normalized_runtime_fields'] == recipe['runtime_fields']
             and holdout['returncode'] == original['returncode'] == 0,
             'original observation does not match the qualified source and CPU domain')
    _require(len(argv) >= 2 and all(type(arg) is str and '{output_dir}' not in arg for arg in argv)
             and argv[-2] == flag and argv.count(flag) == 1
             and leaf not in ('', '.', '..') and Path(leaf).name == leaf
             and set(holdout['files']) == {'out'}
             and Path(holdout['files']['out']['uri']) == cwd / leaf
             and original['published'] == holdout['files']['out']
             and original['stdout'] == holdout['stdout'] and original['argv'] == argv
             and original['command'] == holdout['command']
             == ['/usr/bin/taskset', '-c', ','.join(map(str, cpus)), interpreter['uri'], '-B', source['uri']],
             'expected an exact original argv with one terminal publication in the captured working directory')
    executable_path = Path(runtime['uri']).parent / package['binary']
    _require(kept['reused_binary'] == executable and Path(executable['uri']) == executable_path
             and executable_path.resolve(strict=True) == executable_path
             and executable_path.is_relative_to(Path(runtime['uri']).parent)
             and read_artifact(executable, json_document=False).startswith(b'\x7fELF')
             and os.access(executable_path, os.X_OK) and os.access(interpreter['uri'], os.X_OK)
             and package['cxx_standard'] == 20 and package['python_runtime_required'] is False
             and not package.get('launch_prefix') and package['deployment_authorized'] is False
             and kept['action'] == 'RETAIN_VERIFIED_LOCAL_BINARY'
             and kept['deployment_authorized'] is False,
             'expected the retained independent local C++20 ELF')
    read_artifact(interpreter, json_document=False)
    invocation = dict(executable=executable, original_executable=interpreter, cpu_ids=cpus, cwd=str(cwd),
                      original_environment=environment, environment=dict(environment, **package['launch_environment']),
                      output_index=len(argv) - 1)
    return source, dict(retention=reference, runtime=runtime, holdout=holdout_ref, recipe=recipe_ref,
                        argv=argv, invocation=invocation, normalization=recipe['runtime_fields'])


def _authority(entry, profiles):
    _require(type(profiles) is list and profiles and all(
        type(row) is dict and type(row.get('argv')) is list and all(type(arg) is str for arg in row['argv'])
        for row in profiles
    ) and len({tuple(row['argv']) for row in profiles}) == len(profiles),
             'an entry requires distinct, independently qualified argv profiles')
    return dict(schema=SCHEMA, schema_version=1, entry=entry, profiles=profiles, python_runtime_required=False,
                deployment_authorized=False)


def register(retained_paths, *, out):
    profiles = []
    source = None
    for path in retained_paths:
        entry, profile = _profile(_ref(path, retention.SCHEMA))
        _require(source in (None, entry), 'all profiles must identify the same source path and bytes')
        source = entry
        profiles.append(profile)
    authority = _authority(source, profiles)
    destination = Path(out).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write(destination, authority)
    return _ref(destination, SCHEMA)


def _read(reference):
    authority = read_artifact(reference)
    entry, profiles = authority['entry'], authority['profiles']
    _require(authority == _authority(entry, profiles) and entry == _ref(entry['uri'], _SOURCE),
             'source authority envelope differs')
    return entry, profiles


def _verify_profile(source, profile):
    entry, recovered = _profile(profile['retention'])
    _require(entry == source and profile == recovered, 'source authority profile differs from cold replay')
    return recovered


def check(path):
    reference = _ref(path, SCHEMA)
    entry, profiles = _read(reference)
    return _authority(entry, [_verify_profile(entry, profile) for profile in profiles])


def _result(provider, source, argv):
    entry, profiles = _read(provider)
    matching = [row for row in profiles if row['argv'] == argv]
    _require(source == entry and len(matching) == 1,
             'source path, content, and exact original argv require one qualified profile')
    profile = _verify_profile(source, matching[0])
    return dict(schema=RESULT_SCHEMA, schema_version=1, provider=provider, entry=source, **profile,
                runtime_reused=True, executables_generated=0, python_runtime_required=False,
                deployment_authorized=False)


def compile_entry(provider_path, entry, argv, out):
    requested = Path(out).absolute()
    _require(not requested.exists() and not requested.is_symlink(), 'require a fresh compilation path')
    result = _result(_ref(provider_path, SCHEMA), _ref(entry, _SOURCE), list(argv))
    requested.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-array-task-', dir=requested.parent) as temporary:
        staging = Path(temporary) / 'entry'
        staging.mkdir()
        _write(staging / _RECEIPT, result)
        staging.rename(requested)
    return _ref(requested / _RECEIPT, RESULT_SCHEMA)


def check_entry(path):
    value = read_artifact(_ref(path, RESULT_SCHEMA))
    recovered = _result(value['provider'], value['entry'], value['argv'])
    _require(value == recovered, 'compilation receipt differs from cold replay')
    return recovered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retained', action='append', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args(argv)
    if args.check and not any((args.retained, args.out)):
        result = check(args.check)
    elif args.retained and args.out and args.check is None:
        result = register(args.retained, out=args.out)
    else:
        parser.error('provide --retained RETENTION [--retained RETENTION ...] --out PROVIDER or --check PROVIDER')
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()

"""Recover clean replay boundaries from already admitted native CPU entries."""

import json
import os
from pathlib import Path

from scar.ir.execution_recipe import read_artifact


def _require(valid, message):
    if not valid:
        raise ValueError(message)


def _member(manifest, member):
    relative = Path(member['uri'])
    root = Path(manifest['uri']).parent
    path = root / relative
    _require(not relative.is_absolute() and relative.name and '..' not in relative.parts
             and relative.as_posix() == member['uri'] and path.resolve(strict=True) == path,
             'CPU replay executable must be a local package member')
    return {**member, 'uri': str(path)}


def _arm(executable, arguments, environment, stdout):
    read_artifact(executable, json_document=False)
    _require(os.access(executable['uri'], os.X_OK), 'CPU replay executable is not executable')
    read_artifact(stdout, json_document=False)
    return dict(argv=[executable['uri'], *arguments], environment=dict(environment),
                stdout=stdout, executable=executable)


def _scope(record, compiled, authority):
    _require(record['entry'] == compiled['entry'] == authority['entry']
             and record['argv'] == compiled['argv'],
             'CPU replay requested a different original source or argv')


def _match_json_report(stdout, published, recipe, expected):
    from scar.compiler.calibrate_array_entry import _normalize

    def unique_fields(pairs):
        document = dict(pairs)
        _require(len(document) == len(pairs), 'array task report has duplicate JSON fields')
        return document

    def canonical(raw):
        document = json.loads(raw, object_pairs_hook=unique_fields)
        return json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False)

    return canonical(stdout) == canonical(published) and _normalize(stdout, recipe) == expected


def sparse(record, compiled, authority):
    """Replay the sealed count, observed original stdout and selected native package."""
    _scope(record, compiled, authority)
    profile = read_artifact(compiled['compilation'])
    benchmark = read_artifact(profile['benchmark'])
    launch = benchmark['launch']
    decision = read_artifact(profile['decision'])
    selected = decision['selected_package']
    package = read_artifact(selected)
    stdout = profile['oracle']
    interpreter = authority['interpreter_executable']
    _require(compiled['family'] == record['provider']
             and launch['entry'] == record['entry'] and launch['argv'] == record['argv']
             and launch['interpreter'] == interpreter['uri']
             and launch['stdin'] == 'DEVNULL'
             and selected == profile['packages'][str(profile['selected_native_width'])]
             and package['fixed_argv'] == record['argv']
             and set(package['cpu_ids']) <= set(launch['cpu_ids'])
             and decision['exact_stdout_sha256'] == stdout['sha256'],
             'sparse replay scope differs from the sealed count and clean comparison')
    environment = launch['environment']
    return dict(cpu_ids=launch['cpu_ids'], cwd=launch['cwd'],
                arms={
                    'original': _arm(interpreter, ['-B', record['entry']['uri'], *record['argv']],
                                     environment, stdout),
                    'candidate': _arm(_member(selected, package['executable']), record['argv'],
                                      environment, stdout),
                },
                files={}, transcript=[])


def fixed_grid(record, compiled, authority):
    """Replay both distinct stdout contracts and the complete final float32 file."""
    _scope(record, compiled, authority)
    invocation, validation = compiled['invocation'], compiled['validation']
    package_ref = compiled['runtime']
    package = read_artifact(package_ref)
    executable_path = package['executable']
    executable = _member(package_ref, dict(schema='application/x-executable', uri=executable_path,
                                           **package['files'][executable_path]))
    original, native = validation['original'], validation['native']
    input_dir, step_count, output_path = record['argv']
    _require(compiled['provider'] == record['provider'] and invocation['package'] == package_ref
             and invocation['executable'] == executable_path
             and invocation['arguments'] == [step_count, *(ref['uri'] for ref in authority['inputs']), output_path]
             and validation['bitwise_final_tensor'] is True
             and validation['stdout_contract'] == 'SOURCE_STATUS_AND_NATIVE_TENSOR_ABI_METADATA_CHECKED_SEPARATELY'
             and read_artifact(original['output'], json_document=False)
                 == read_artifact(native['output'], json_document=False),
             'fixed-grid replay scope differs from the verified file and stdout contracts')
    context = authority['context']
    native_environment = invocation['environment']
    original_environment = dict(native_environment, PYTHONPATH=':'.join(context['pythonpath']))
    output_name = 'final.f32'
    output_argument = '{output_dir}/' + output_name
    return dict(cpu_ids=invocation['cpus'], cwd=context['cwd'],
                arms={
                    'original': _arm(context['interpreter'], ['-B', record['entry']['uri'], input_dir, step_count,
                                                               output_argument], original_environment, original['stdout']),
                    'candidate': _arm(executable, [*invocation['arguments'][:-1], output_argument],
                                      native_environment, native['stdout']),
                },
                files={output_name: original['output']}, transcript=[])


def array_tasks(record, compiled, authority):
    """Replay a retained array executable against its independently observed report."""
    _scope(record, compiled, authority)
    invocation = compiled['invocation']
    runtime = read_artifact(compiled['runtime'])
    recipe = read_artifact(compiled['recipe'])
    holdout = read_artifact(compiled['holdout'])
    argv = list(record['argv'])
    index = invocation['output_index']
    flags = recipe['options'][recipe['output_option']]['flags']
    original = invocation['original_executable']
    native = invocation['executable']
    _require(type(index) is int and index == len(argv) - 1 and index >= 1
             and argv[index - 1] in flags and argv.count(argv[index - 1]) == 1,
             'array task replay requires one qualified terminal output option')
    name = argv[index]
    _require(Path(name).name == name and name not in ('', '.', '..') and '{output_dir}' not in name,
             'array task publication must be a relative leaf')
    cpus = invocation['cpu_ids']
    original_prefix = [original['uri'], '-B', record['entry']['uri']]
    _require(compiled['provider'] == record['provider']
             and compiled['recipe'] == runtime['recipe'] == holdout['recipe']
             and record['entry'] == recipe['entry'] == holdout['entry'] and record['argv'] == holdout['argv']
             and original == recipe['source_python'] == holdout['interpreter']
             and native == runtime['files'][runtime['binary']]
             and cpus == runtime['cpu_ids'] == holdout['cpu_ids']
             and holdout['command'] == ['/usr/bin/taskset', '-c', ','.join(map(str, cpus)), *original_prefix]
             and invocation['original_environment'] == recipe['environment'] == holdout['environment']
             and invocation['environment'] == dict(recipe['environment'], **runtime.get('launch_environment', {}))
             and compiled['normalization'] == recipe['runtime_fields'] == holdout['normalized_runtime_fields'],
             'array task replay differs from its qualified source, launch or report contract')
    output = holdout['files'][recipe['output_option']]
    observed = read_artifact(holdout['stdout'], json_document=False)
    published = read_artifact(output, json_document=False)
    expected = read_artifact(holdout['normalized'], json_document=False)
    _require(_match_json_report(observed, published, recipe, expected),
             'array task holdout stdout and publication differ from the independently observed report')
    argv[index] = '{output_dir}/' + name
    return dict(cpu_ids=cpus, cwd=invocation['cwd'],
                arms={
                    'original': dict(_arm(original, [*original_prefix[1:], *argv],
                                          invocation['original_environment'], holdout['stdout']), output_slot=index + 3),
                    'candidate': dict(_arm(native, argv, invocation['environment'], holdout['stdout']),
                                      output_slot=index + 1),
                },
                files={name: output}, transcript=[],
                json_report=dict(runtime_fields=compiled['normalization'], oracle=holdout['normalized']))

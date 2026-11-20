"""Recover clean replay boundaries from already admitted native CPU entries."""

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

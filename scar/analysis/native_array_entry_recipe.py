"""Admit original invocation/report ports for finite checked numerical profiles."""

import argparse
import json
from pathlib import Path

from scar.codegen.native_array_entry import RECIPE, _FLOAT_FIELDS, _INTEGER_FIELDS, _KINDS, _RUNTIME_FIELDS
from scar.ir.execution_recipe import artifact_reference, read_artifact


ROLES = 'scar.array-entry-report-ports'
_ORIGINAL = 'scar.array-entry-original-process'


def _need(ok, reason):
    if not ok:
        raise ValueError('array entry recipe: ' + reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _at(document, path):
    value = document
    for key in path:
        value = value[key]
    return value


def _json_file(ref):
    return json.loads(read_artifact(ref, json_document=False))


def _matches(witness, native):
    report = _json_file(witness['stdout'])
    prior = read_artifact(witness['earlier_trajectory_run'])
    original = prior['trajectory']['records']
    candidate = native['checkpoint_validation']
    return (prior['execution']['entry_path'] == witness['entry']['uri']
            and len(original) == len(candidate) and all(all(expected['output'][key] == observed[key]
                    for key in ('size_bytes', 'sha256')) for expected, observed in zip(original, candidate))
            and report['parameters'] == witness['parameters'])


def _options(contract):
    original = contract['argv']
    _need(original['parse_method'] == 'parse_args', 'only strict original named arguments are admitted')
    values = {}
    for option in original['options']:
        kind = option['kind']
        _need(kind in _KINDS and option['action'] == 'store', 'unsupported original operand type')
        initial = option['default']['value']
        values[option['id']] = dict(kind=kind, default=initial, flags=option['flags'],
                                    choices=[item['value'] for item in option['choices']])
    return values


def _profiles(refs, witnesses, options, routes, report_layout):
    common = []
    profiles = []
    observed = [(ref, read_artifact(ref)) for ref in witnesses]
    tolerance_key, output_key = routes['tolerance_option'], routes['output_option']
    fields = {tuple(row['path']): row['role'] for row in routes['report_bindings']}
    expected_norm = next(path for path, role in fields.items() if role == 'norm')
    runtime_paths = set()
    for path in report_layout['volatile_paths']:
        runtime_paths.add(tuple(0 if key == '*' else key for key in path))
    _need(runtime_paths == {path for path, role in fields.items() if role in _RUNTIME_FIELDS},
          'publication routes differ from original runtime fields')
    for ref in refs:
        native = read_artifact(ref)
        matches = [(source_ref, witness) for source_ref, witness in observed if _matches(witness, native)]
        _need(len(matches) == 1 and native['python_runtime_required'] is False
              and native['deployment_authorized'] is False, 'native profile has no unique original run')
        witness_ref, witness = matches[0]
        report = _json_file(witness['stdout'])
        _need(len(report[report_layout['results_field']]) == 1
              and report == _json_file(witness['file_outputs'][output_key])
              and witness['run_returncode'] == 0 and _at(report, expected_norm) > 0,
              'original output is outside the single-worker report profile')
        reported = report['parameters']
        _need(set(reported) | {output_key} == set(options), 'original parameters differ from source declaration')
        selection = {key: value for key, value in reported.items() if key != tolerance_key}
        for key, value in selection.items():
            choices = options[key]['choices']
            _need(not choices or value in choices, 'profile violates original argparse choices')
        supplied_key = next(path for path, role in fields.items() if role == 'optional_tolerance')
        default_key = next(path for path, role in fields.items() if role == 'tolerance')
        default = _at(report, default_key)
        _need(_at(report, supplied_key) is None and type(default) is float and 0 < default < 1,
              'default tolerance must come from an original defaulted invocation')
        for path, role in fields.items():
            value = _at(report, path)
            _need((role in _INTEGER_FIELDS and type(value) is int)
                  or (role in _FLOAT_FIELDS and type(value) is float)
                  or (role == 'pid_list' and type(value) is list and len(value) == 1 and type(value[0]) is int)
                  or (role == 'optional_tolerance' and value is None), 'report binding type differs')
        common.append((native['normalize'], native['cpu_ids'], native['stock_provider']['provider'],
                       witness['python_executable'], witness['environment']))
        profiles.append(dict(package=ref, witness=witness_ref, tape_sha256=native['tape']['sha256'],
            parameters=selection, report=report, default_tolerance=default, original_argv=witness['argv']))
    _need(len(profiles) > 0 and len({json.dumps(row['parameters'], sort_keys=True) for row in profiles}) == len(profiles)
          and all(row == common[0] for row in common), 'profile selectors overlap or share different providers')
    return profiles, common[0]


def project(contract_path, roles_path, packages):
    from scar.analysis import source_array_entry_contract as source

    authority = _ref(contract_path, source.SCHEMA)
    contract = source.check(authority['uri'])
    _need(contract == read_artifact(authority), 'original entry contract changed')
    role_ref = _ref(roles_path, ROLES)
    roles = read_artifact(role_ref)
    _need(roles['schema_version'] == 1 and roles['source_contract'] == authority,
          'report ports require the exact original static contract')
    layout = contract['report']
    _need(layout['path_output']['option'] == roles['output_option'] and layout['path_output']['indent'] == 2,
          'original file effect differs from the native renderer')
    options = _options(contract)
    _need(options[roles['output_option']]['kind'] == 'path'
          and options[roles['tolerance_option']]['kind'] == 'float64'
          and options[roles['tolerance_option']]['default'] is None,
          'original output path or tolerance type differs')
    refs = [_ref(path, 'scar.native-array-tape') for path in packages]
    profiles, (normalize, cpus, _, python, environment) = _profiles(
        refs, [row['observation'] for row in contract['witnesses']], options, roles, layout)
    read_artifact(python, json_document=False)
    return dict(schema=RECIPE, schema_version=1, source_contract=authority, roles=role_ref,
        entry=contract['entry'], options=options, allow_abbrev=True, profiles=profiles,
        report_bindings=roles['report_bindings'], tolerance_option=roles['tolerance_option'],
        output_option=roles['output_option'], normalize=normalize, cpu_ids=cpus,
        source_python=python, environment=environment, runtime_fields=layout['volatile_paths'],
        source_metadata_policy='PRESERVE_AUTHORED_BACKEND_LITERAL_AND_QUALIFIED_SOURCE_INTERPRETER',
        unrecognized_profile='DECLINE_BEFORE_NUMERICAL_OR_PUBLICATION_WORK',
        deployment_authorized=False)


def check(path):
    ref = _ref(path, RECIPE)
    data = read_artifact(ref)
    actual = project(data['source_contract']['uri'], data['roles']['uri'],
                     [row['package']['uri'] for row in data['profiles']])
    _need(actual == data, 'retained recipe differs from original entry and operator evidence')
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--roles', required=True)
    parser.add_argument('--package', action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    result = project(args.contract, args.roles, args.package)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(compiled_profiles=len(result['profiles']), cpu_ids=result['cpu_ids'])))


if __name__ == '__main__':
    main()

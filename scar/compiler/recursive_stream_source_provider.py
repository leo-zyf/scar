"""Register and recompile a source-bound, paced native CPU command stream."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scar.codegen import scalar_recursive_driver as driver_library
from scar.codegen import scalar_recursive_parameterized as native
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import scalar_recursive_parameterized_plan as planning

SCHEMA = 'scar.cpu-recursive-stream-source-provider'
RESULT_SCHEMA = 'scar.cpu-recursive-stream-entry-compilation'
SOURCE_SCHEMA = 'scar.python-source'
_RUNTIME = Path('runtime/artifact.json')


def _require(valid, detail):
    if not valid:
        raise ValueError('native recursive stream: ' + detail)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def _member(root, member):
    relative = Path(member['uri'])
    location = root / relative
    _require(not relative.is_absolute() and location.resolve(strict=True) == location and location.is_relative_to(root),
             'native member path is outside its own package')
    reference = {**member, 'uri': str(location)}
    return reference, read_artifact(reference, json_document=False)


def _authority(seed, *, replay_seed=False):
    artifact = read_artifact(seed)
    _require(seed['schema'] == native.SCHEMA and artifact['schema_version'] == 1,
             'unsupported registered native artifact')
    if replay_seed:
        _require(artifact == native.check(seed['uri']), 'registered native artifact differs from its checked material')
    root = Path(seed['uri']).parent
    _, executable = _member(root, artifact['binary'])
    _require(artifact['binary']['schema'] == 'scar.elf' and executable.startswith(b'\x7fELF'),
             'registered seed binary is not a pinned ELF')
    for name, member in artifact['members'].items():
        _require(member['uri'] == name, 'registered seed member path differs')
        _member(root, member)

    plan_ref = artifact['parameterized_plan']
    driver_ref = artifact['driver_library']
    # Activation only follows a full replay: seed validation for authority
    # publication, or the exporter / detached runtime check for entry builds.
    plan = read_artifact(plan_ref)
    driver = read_artifact(driver_ref)
    scope = plan['scope']
    entry = _ref(scope['source']['uri'], SOURCE_SCHEMA)
    _require(plan_ref['schema'] == planning.SCHEMA and scope['mode'] == planning.MODE and
             driver_ref['schema'] == driver_library.SCHEMA and driver['schema_version'] == 2 and
             driver['code'] == plan['ports']['driver']['code'] and
             driver['abi']['control'] == 'ONE_WORLD_ACROSS_REBOUND_SOURCE_GENERATORS' and
             entry == scope['source'] == artifact['source'] and
             plan['decision']['native_experiment_authorized'] is True and
             plan['decision']['deployment_authorized'] is False and
             artifact['cxx_standard'] == 20 and artifact['python_runtime_required'] is False and
             artifact['deployment_authorized'] is False and artifact['exact_command_mismatch_status'] == 77 and
             artifact['consumes_runtime_command_stream'] is True, 'expected the scoped C++20 live-stream profile')

    initial = read_artifact(scope['entry_plan'])
    source_entry = read_artifact(initial['artifacts']['source_entry'])
    capture_ref = source_entry['capture']
    capture = read_artifact(capture_ref)
    launch = capture['launch']
    _require(source_entry['source'] == capture['source'] == entry and
             capture['original_execution'] is True and capture['performance_measurement'] is False and
             launch['entry_path'] == entry['uri'] and launch['exit_code'] == 0 and launch['argv'] == [],
             'original entry has no matching empty-argv observation')

    grammar = plan['grammar']
    _require(plan['requirements']['paced_only'] is True and
             plan['requirements']['parse_before_effects'] is True and
             plan['requirements']['held_out_input_not_embedded'] is True and
             plan['resources']['cpu_owner_lanes'] == 1 and
             plan['resources']['concurrent_command_effects'] is False,
             'the checked native authority must use one paced command owner')
    commands = [{key: row[key] for key in ('kind', 'tokens', 'arguments', 'optional_tail')}
                for row in grammar['commands']]
    stdin = dict(kind='LIVE_STDIN', encoding='ASCII', protocol=grammar['protocol'],
                 tokenization=grammar['tokenization'], commands=commands, depth_range=grammar['depth_range'],
                 publication=plan['source_effects']['acknowledgement'],
                 unsupported_command=grammar['malformed_or_unsupported'],
                 decline_status=artifact['exact_command_mismatch_status'])
    authority = dict(schema=SCHEMA, schema_version=1, entry=entry, argv=[],
                     source_artifact=seed, parameterized_plan=plan_ref, driver_library=driver_ref,
                     original_launch=capture_ref, scope=scope['mode'], stdin=stdin,
                     normalization=scope['normalization'], unbound=plan['proof_frontiers']['unbound'],
                     runtime='LINUX_CXX20_CPU', python_runtime_required=False, deployment_authorized=False)
    return authority, artifact


def register(artifact, *, out):
    value, _ = _authority(_ref(artifact, native.SCHEMA), replay_seed=True)
    out = Path(out).absolute()
    out.parent.mkdir(parents=True, exist_ok=True)
    _write(out, value)
    return _ref(out, SCHEMA)


def _checked_provider(reference, *, replay_seed=False):
    _require(reference['schema'] == SCHEMA, 'result has another provider type')
    value = read_artifact(reference)
    authority, seed = _authority(value['source_artifact'], replay_seed=replay_seed)
    _require(value == authority, 'provider receipt differs from cold replay')
    return value, seed


def check(path):
    value, _ = _checked_provider(_ref(path, SCHEMA), replay_seed=True)
    return value


def _invocation(runtime, artifact, provider, seed, source, argv):
    _require(artifact['parameterized_plan'] == provider['parameterized_plan'] and
             artifact['driver_library'] == provider['driver_library'] and artifact['source'] == source and
             artifact['schema_version'] == seed['schema_version'] == 1 and
             artifact['python_runtime_required'] is False and artifact['cxx_standard'] == 20 and
             artifact['exact_command_mismatch_status'] == provider['stdin']['decline_status'] and
             artifact['source_ports'] == seed['source_ports'] and artifact['members'] == seed['members'] and
             artifact['link_dependencies'] == seed['link_dependencies'] and
             artifact['consumes_runtime_command_stream'] is True and artifact['deployment_authorized'] is False,
             'detached product or ABI scope differs from the qualified source provider')
    root = Path(runtime['uri']).parent
    executable, data = _member(root, artifact['binary'])
    _require(executable['schema'] == 'scar.elf' and data.startswith(b'\x7fELF'),
             'detached binary does not match its package-local reference')
    return dict(executable=executable, argv=argv, stdin=provider['stdin'], runtime=provider['runtime'])


def _receipt(provider, source, argv, authority, runtime, invocation):
    return dict(schema=RESULT_SCHEMA, schema_version=1, provider=provider, entry=source, argv=argv,
                scope=authority['scope'], runtime=runtime, invocation=invocation,
                normalization=authority['normalization'], unbound=authority['unbound'],
                validation='CHECKED_SOURCE_PROFILE_AND_REBUILT_NATIVE_RUNTIME',
                python_runtime_required=False, deployment_authorized=False)


def compile_entry(provider_path, entry, argv, out):
    argv = list(argv)
    _require(argv == [], 'the registered stream uses an empty original argv')
    provider = _ref(provider_path, SCHEMA)
    authority, seed = _checked_provider(provider)
    source = _ref(entry, SOURCE_SCHEMA)
    _require(source == authority['entry'] and argv == authority['argv'] == [],
             'requested source path, bytes or argv differs from the registered profile')

    requested = Path(out).absolute()
    _require(not requested.exists() and not requested.is_symlink(), 'require a fresh compilation path')
    requested.parent.mkdir(parents=True, exist_ok=True)
    out = requested.parent.resolve(strict=True) / requested.name
    with TemporaryDirectory(prefix='.scar-recursive-stream-', dir=out.parent) as temporary:
        package = Path(temporary) / 'product'
        package.mkdir()
        exported = native.export(authority['parameterized_plan']['uri'], authority['driver_library']['uri'],
                                 package / _RUNTIME.parent)
        runtime = _ref(package / _RUNTIME, native.SCHEMA)
        _require(exported == read_artifact(runtime), 'detached native build receipt differs')
        invocation = _invocation(runtime, exported, authority, seed, source, argv)
        local_binary = Path(invocation['executable']['uri']).relative_to(package)
        invocation['executable']['uri'] = str(out / local_binary)
        runtime['uri'] = str(out / _RUNTIME)
        _write(package / 'result.json', _receipt(provider, source, argv, authority, runtime, invocation))
        package.rename(out)
    return _ref(out / 'result.json', RESULT_SCHEMA)


def check_entry(path):
    reference = _ref(path, RESULT_SCHEMA)
    value = read_artifact(reference)
    provider, source, argv = (value[name] for name in ('provider', 'entry', 'argv'))
    authority, seed = _checked_provider(provider)
    _require(source == authority['entry'] and argv == authority['argv'] == [],
             'result does not match the provider source and invocation')
    runtime = _ref(Path(reference['uri']).parent / _RUNTIME, native.SCHEMA)
    _require(runtime == value['runtime'], 'result must reference its own detached runtime')
    artifact = native.check(runtime['uri'])
    _require(artifact == read_artifact(runtime), 'detached runtime changed during replay')
    invocation = _invocation(runtime, artifact, authority, seed, source, argv)
    _require(value == _receipt(provider, source, argv, authority, runtime, invocation),
             'compilation receipt differs from cold replay')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifact', 'out', 'check'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    if args.check and not any((args.artifact, args.out)):
        probe = _ref(args.check, 'scar.recursive-stream-document')
        schema = json.loads(read_artifact(probe, json_document=False))['schema']
        result = check_entry(args.check) if schema == RESULT_SCHEMA else check(args.check)
    elif args.artifact and args.out and not args.check:
        result = register(args.artifact, out=args.out)
    else:
        parser.error('provide --check RECEIPT or --artifact NATIVE_ARTIFACT --out PROVIDER')
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()

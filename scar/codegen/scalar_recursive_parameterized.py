"""Export a typed recurring CPU command protocol as a standalone C++20 program."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen import scalar_recursive_entry as entry
from scar.codegen import scalar_recursive_driver as provider
from scar.codegen.native_project import build_native_project, write_native_project
from scar.ir.execution_recipe import read_artifact
from scar.optimization.scalar_score_ir import _int64, _require


SCHEMA = 'scar.scalar-recursive-parameterized-executable'
PLAN = 'scar.scalar-recursive-parameterized-plan'
OBSERVATION = 'scar.scalar-recursive-parameterized-observation'
_PROGRAM = 'scar_recursive_parameterized'
_BINARY = 'bin/' + _PROGRAM
_TEMPLATE = Path(__file__).with_name('scalar_recursive_parameterized.cpp.in')


def _roles(recipe, prior, native):
    grammar = recipe['grammar']
    commands = {row['kind']:row['tokens'] for row in grammar['commands']}
    _require(set(commands) == {'EMPTY','POSITION_START','POSITION_FEN','GO_DEPTH','UCI','ISREADY','NEW_GAME','QUIT'}
             and commands['EMPTY'] == [] and len(commands['POSITION_START']) == len(commands['POSITION_FEN']) == 2
             and commands['POSITION_START'][0] == commands['POSITION_FEN'][0] and len(commands['GO_DEPTH']) == 2
             and all(len(commands[role]) == 1 for role in ('UCI','ISREADY','NEW_GAME','QUIT')),
             'parameterized command shape differs from the typed source dispatch')
    position, start = commands['POSITION_START']
    go, depth = commands['GO_DEPTH']
    ident, ready, reset, quit_ = (commands[role][0] for role in ('UCI','ISREADY','NEW_GAME','QUIT'))
    replies = {row['line']:row['reply_lines'] for row in prior['commands'] if row['operation'] == 'STATIC_REPLY'}
    _require(set(replies) == {ident, ready}, 'the immutable reply ports do not match the source grammar')
    clock = recipe['configuration']['clocks']
    domain = grammar['depth_range']
    defaults = tuple(recipe['configuration']['values'][port] for port in native['configuration_ports'])
    _require(all(_int64(value) for value in defaults) and len(defaults) == len(recipe['configuration']['values'])
             and 0 < domain['minimum'] <= domain['maximum'] < 1000,
             'native depth or scalar configuration is not admitted')
    ports = dict(POSITION=position, START=start, GO=go, DEPTH=depth, IDENT=ident, READY=ready, RESET=reset, QUIT=quit_,
                 IDENT_REPLY=''.join(text + '\n' for text in replies[ident]),
                 READY_REPLY=''.join(text + '\n' for text in replies[ready]))
    output = {key:entry._literal_string(value) for key, value in ports.items()}
    output.update(MIN_DEPTH=str(domain['minimum']), MAX_DEPTH=str(domain['maximum']),
                  BUDGET=repr(float(clock['default_budget_seconds'])), HARD=repr(float(clock['hard_cap_seconds'])),
                  SOFT=repr(float(clock['soft_initial_epoch'])),
                  CONFIG_COUNT=str(len(defaults)), CONFIG=', '.join(map(str, defaults)))
    return output


def _render(recipe, prior, driver, ui, native):
    common, boundary, controller = entry._FILE.read_text().partition('\nint main(')
    _require(boundary and controller and common.count('@SCAR_TAPE@') == 1, 'checked search host cannot be isolated')
    fields, _ = entry._consumer(ui)
    state_ports = dict(MOVE_MAP=('yield','move_lookup'), SCORE_MAP=('initialization','score_table'),
                       HISTORY=('initialization','history'), ROOT=('initialization','root'))
    for key, (role, action) in state_ports.items():
        fields[key] = native['world'][driver[role][action]['field']]['field']
    fields.update(TAPE='', **_roles(recipe, prior, native))
    source = common + '\n' + _TEMPLATE.read_text()
    for key, value in fields.items():
        source = source.replace('@SCAR_' + key + '@', value)
    _require('@SCAR_' not in source, 'unbound C++ command or world port')
    return source


def _materials(plan_path, native_path):
    from scar.codegen import scalar_recursive_source_position, scalar_recursive_source_pv
    from scar.optimization import scalar_recursive_parameterized_plan

    plan_ref = entry._ref(plan_path, PLAN)
    library_ref = entry._ref(native_path, provider.SCHEMA)
    recipe = scalar_recursive_parameterized_plan.check(plan_ref['uri'])
    native = provider.check(library_ref['uri'])
    _require(recipe['decision']['native_experiment_authorized'] is True
             and recipe['decision']['deployment_authorized'] is False and native['schema_version'] == 2,
             'parameterized release needs the proved source protocol and v2 owner')
    scope = recipe['scope']
    prior = read_artifact(scope['entry_plan'])
    driver, ui = (read_artifact(scope[role]) for role in ('source_driver','ui_contract'))
    _require(driver['source'] == prior['scope']['source'] == scope['source']
             and native['code'] == driver['driver']['code'], 'native driver source identity differs')
    render_args = (scope['ui_contract']['uri'], library_ref['uri'])
    pv = scalar_recursive_source_pv.render(*render_args)
    position = scalar_recursive_source_position.render(*render_args)
    host = _render(recipe, prior, driver, ui, native)
    root = Path(library_ref['uri']).parent
    members = {}
    for name, reference in native['members'].items():
        if name.startswith('include/') or name == 'src/driver.cpp':
            location = (root / name).resolve(strict=True)
            _require(location.is_relative_to(root) and reference['uri'] == name, 'invalid provider source reference')
            members[name] = read_artifact(dict(reference, uri=str(location)), json_document=False)
    kernel = members.pop('src/driver.cpp')
    members.update({'src/main.cpp': kernel + b'\n' + host.encode('ascii'),
                    'src/source_pv.inc': pv['source'].encode('ascii'),
                    'src/source_position.inc': position['source'].encode('ascii')})
    ports = dict(position=position['source_ports'], output=pv['source_ports'])
    return plan_ref, library_ref, recipe, members, ports


def _project(root):
    write_native_project(root, _PROGRAM, include_dirs=('include',), compile_options=entry._FLAGS)


def _member_refs(root, members):
    return {key:entry._local(root, key, 'application/octet-stream') for key in sorted([*members,'CMakeLists.txt'])}


def export(plan_path, native_path, out):
    plan, library, recipe, members, ports = _materials(plan_path, native_path)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    _require(compiler, 'an available C++20 compiler is required')
    with TemporaryDirectory(prefix='.scar-parameterized-entry-', dir=out.parent) as temporary:
        root = Path(temporary) / 'package'
        root.mkdir()
        for key, raw in members.items():
            path = root / key
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(raw)
        _project(root)
        binary, _ = build_native_project(root, _PROGRAM, compiler)
        deps = entry._elf(binary)
        denied = subprocess.run([str(binary)], cwd='/', env={}, input=b'\0\n', capture_output=True, timeout=10)
        _require(denied.returncode == 77 and not denied.stdout and not denied.stderr, 'unadmitted command published output')
        artifact = dict(schema=SCHEMA, schema_version=1, parameterized_plan=plan, driver_library=library,
                        source=recipe['scope']['source'], binary=entry._local(root, _BINARY, 'scar.elf'),
                        cxx_standard=20, python_runtime_required=False, deployment_authorized=False,
                        compiler=dict(command=compiler, version=entry._version(compiler)), link_dependencies=deps,
                        source_ports=ports, exact_command_mismatch_status=77, members=_member_refs(root, members),
                        consumes_runtime_command_stream=True)
        (root / 'artifact.json').write_text(json.dumps(artifact, indent=2, sort_keys=True) + '\n')
        root.rename(out)
    return artifact


def check(path):
    reference = entry._ref(path, SCHEMA)
    result = read_artifact(reference)
    root = Path(reference['uri']).parent
    plan, library, recipe, members, ports = _materials(result['parameterized_plan']['uri'], result['driver_library']['uri'])
    _require(result['schema_version'] == 1 and result['parameterized_plan'] == plan and result['driver_library'] == library
             and result['source'] == recipe['scope']['source'] and result['binary'] == entry._local(root, _BINARY, 'scar.elf')
             and result['source_ports'] == ports and result['cxx_standard'] == 20
             and result['python_runtime_required'] is result['deployment_authorized'] is False
             and result['consumes_runtime_command_stream'] is True and result['exact_command_mismatch_status'] == 77
             and result['link_dependencies'] == entry._elf(root / _BINARY)
             and result['members'] == _member_refs(root, members)
             and all((root / key).read_bytes() == data for key, data in members.items()),
             'stored parameterized executable differs from typed source and build inputs')
    with TemporaryDirectory(prefix='scar-parameterized-check-') as temporary:
        _project(Path(temporary))
        _require((Path(temporary) / 'CMakeLists.txt').read_bytes() == (root / 'CMakeLists.txt').read_bytes(),
                 'parameterized build instructions differ')
    return result


def project_observation(artifact_path, directory, original_capture_path):
    """Replay live root construction, halfmove state and raw recursive results."""
    from scar.analysis.scalar_recursive_driver import CAPTURE
    from scar.codegen.scalar_recursive_standalone import _encoded, _state

    artifact_ref = entry._ref(artifact_path, SCHEMA)
    artifact = read_artifact(artifact_ref)
    recipe = read_artifact(artifact['parameterized_plan'])
    scope = recipe['scope']
    prior = read_artifact(scope['entry_plan'])
    driver = read_artifact(scope['source_driver'])
    ui = read_artifact(scope['ui_contract'])
    native = read_artifact(artifact['driver_library'])
    _, sorted_plan = entry._sorted(native)
    original_ref = entry._ref(original_capture_path, CAPTURE)
    original = read_artifact(original_ref)
    _require(original['source'] == scope['source'] == driver['source'] and original['original_execution'] is True
             and original['performance_measurement'] is False and not original['frontiers'],
             'held-out source observation has incomplete or different original execution')
    design = read_artifact(original['design'])
    _require(design['driver'] == driver['driver'] and design['recursive_region'] == driver['recursive_region'],
             'held-out source observation belongs to another driver')
    startup = read_artifact(read_artifact(prior['artifacts']['source_entry'])['capture'])
    version = startup['initial_table']
    spelling = next(row['tokens'] for row in recipe['grammar']['commands'] if row['kind']=='GO_DEPTH')
    commands = read_artifact(original['stdin'], json_document=False).decode('ascii').splitlines()
    depths = [int(tokens[2]) for text in commands if (tokens:=text.split())[:2] == spelling and len(tokens)==3]
    sessions = original['sessions']
    _require(depths and len(depths) == len(sessions), 'held-out source observation must cover every requested search')
    result, files = entry._read_trace(directory, driver, native, sorted_plan, sessions)
    root = Path(directory).resolve(strict=True)
    for index, (session, depth) in enumerate(zip(sessions, depths, strict=True)):
        mode = session['consumer_scalars']
        _require(mode[ui['request_port']] == mode[ui['limit_port']] == depth
                 and session['native_configuration'] == recipe['configuration']['values']
                 and session['exit']['kind'] == 'GENERATOR_CLOSE', 'held-out source mode left the intended projection')
        halfmove = mode[recipe['positions']['halfmove_port']]
        table = native['table_slots'].index(version)
        typed_history = [_state(value, sorted_plan) for value in session['history_input']]
        expected = 'SCAR_ENTRY_POSITION_V1\n' + _encoded((table, halfmove, len(typed_history), typed_history))
        prepare = entry._ref(root / f'{index}.prepare', 'application/octet-stream')
        _require(prepare['uri'] == str(root / f'{index}.prepare')
                 and read_artifact(prepare, json_document=False) == expected.encode('ascii'),
                 'native runtime position, halfmove clock or pre-search table epoch differs')
        result[index].update(halfmove_clock=halfmove, history_entries=len(typed_history),
                             prior_table_version=version, requested_depth=depth)
        version = session['table_version']
        files.append(prepare)
    return dict(schema=OBSERVATION, schema_version=1, artifact=artifact_ref, original_capture=original_ref,
                observed_from_exact_launch=True, original_sessions_matched=True, sessions=result, trace_files=files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ('plan','driver-library','out','check'):
        parser.add_argument('--' + role,type=Path)
    args = parser.parse_args(argv)
    supplied = (args.plan, args.driver_library, args.out)
    if args.check and not any(supplied):
        result = check(args.check)
    elif all(supplied) and not args.check:
        result = export(*supplied)
    else:
        parser.error('provide --plan, --driver-library and --out, or --check')
    print(json.dumps({key:result[key] for key in ('schema','binary','link_dependencies','python_runtime_required')},
                     sort_keys=True))


if __name__ == '__main__':
    main()

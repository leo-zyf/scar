"""Witness one out-of-domain fixed-grid publication case without changing a user output."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import subprocess

from scar.codegen.cpu_fixed_grid_native_route_package import (
    _output_guards, prepare_cpu_fixed_grid_native_route_launch, verify_cpu_fixed_grid_native_route_package,
)
from scar.codegen.cpu_catalog_runtime_package import _signature
from scar.codegen.native_provider_bundle import _content
from scar.compiler.calibrate_array_entry import _write
from scar.compiler.cpu_fixed_grid_mapped_input_study import _input_files
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.cpu-fixed-grid-publication-guard-study'
_RUN = 'scar.cpu-fixed-grid-publication-guard-run'
_MAPPED, _ORIGINAL = 'NATIVE_CPP', 'ORIGINAL_PYTHON'
_ORDER = (_MAPPED, _ORIGINAL)
_TRACE = Path('/usr/bin/strace')
_RAW = 'application/octet-stream'
_FILE = 'src/runtime/file_io.hpp'
_SENTINEL = b'SCAR private existing-file publication sentinel\n'
_MODE = 0o640


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU fixed-grid publication guard study: ' + detail)


def _raw(reference):
    return read_artifact(reference, json_document=False)


def _ref(path, schema=_RAW):
    _need(path.resolve(strict=True) == path and stat.S_ISREG(path.lstat().st_mode), 'expected regular local evidence')
    return artifact_reference(path, schema)


def _snapshot(path):
    entry = path.lstat()
    _need(stat.S_ISREG(entry.st_mode) and path.resolve(strict=True) == path and entry.st_nlink == 1,
          'private sentinel or retained output must be a canonical, unaliased regular file')
    return dict(device=entry.st_dev, inode=entry.st_ino, mode=stat.S_IMODE(entry.st_mode),
                link_count=entry.st_nlink, **_content(path))


def _source_anchor(reference, token):
    data = _raw(reference)
    _need(data.count(token) == 1, 'expected one admitted publication statement')
    begin = data.index(token)
    end = begin + len(token)
    return dict(source=reference, byte_start=begin, byte_end=end, sha256=sha256(data[begin:end]).hexdigest())


def _context(reference, root):
    package = verify_cpu_fixed_grid_native_route_package(reference)
    _need(package['selected_runtime'] == 'PROVIDER_MAP' and package['inputs_must_remain_stable'] is True
          and package['publication']['precondition'] == 'ABSENT_CXX_EXCLUSIVE_CREATE'
          and package['python_replacement_authorized'] is False, 'mapped route with scoped original default required')
    delivery = read_artifact(package['owned_copy_delivery'])
    original = read_artifact(delivery['observation'])
    route = package['routes']['PROVIDER_MAP']
    plan = read_artifact(package['plan'])
    proof = read_artifact(plan['proof'])
    abi = read_artifact(proof['abi'])
    native = read_artifact(route['project'])
    member = dict(uri=str(Path(route['project_root'])/_FILE), schema='scar.native-cpp-source',
                  **native['files'][_FILE])
    _need(abi['domain']['output'] == 'FRESH_FILE_WITH_EXISTING_PARENT'
          and abi['domain']['normal_completion_only'] is True
          and abi['publication']['order'] == ['FILE', 'STDOUT']
          and proof['scope']['stable_inputs_required_for_full_lifetime'] is True
          and route['project'] == package['selected_project']
          and root.resolve() == root and not root.is_relative_to(Path(reference['uri']).parent)
          and all(not root.is_relative_to(row['project_root']) for row in package['routes'].values())
          and all(not root.is_relative_to(Path(row['path']).parent) for row in package['file_inputs'])
          and package['publication']['output_index'] == len(package['argv']) - 1,
          'probe must keep exact source contract and use a separate private output directory')
    private = root/'isolated-output.f32'
    _need(private != Path(package['publication']['path']), 'study output must not be the user output')
    argv = [*package['argv'][:-1], str(private)]
    command = original['commands'][_ORIGINAL]
    _need(command[-len(package['argv']):] == package['argv'] and original['working_directory'] == package['working_directory']
          and original['python_provider_environment'] == package['source_environment']
          and original['original_cpu_ids'] == package['cpu_ids'],
          'original Python launch changed outside the output argument')
    commands = {_ORIGINAL:[*command[:-1], str(private)], _MAPPED:[route['launcher']['uri'], *argv]}
    source = package['source']
    preflight = _source_anchor(member, b'if (fs::exists(fs::symlink_status(absolute))')
    ingress = _source_anchor(member, b'inputs[i] = at::from_file(')
    exclusive = _source_anchor(member, b'::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666)')
    publication = _source_anchor(member, b'write_fresh_source_file(destinations[0], outputs[0].const_data_ptr<float>(), sizes[0]);')
    _need(preflight['byte_start'] < ingress['byte_start'] < publication['byte_start'],
          'occupied-output check must precede mapped intake and publication')
    return package, commands, private, dict(
        output_arg=dict(index=package['publication']['output_index'], original_unmodified=package['publication']['path'],
                        private=str(private), changed_for_this_study=True),
        fresh_output=dict(package_guard='EXACT_ARGV_AND_ABSENT_OUTPUT_PRELAUNCH',
            generated_early_existence_check=preflight, generated_atomic_exclusive_create=exclusive,
            checked_before_input_ingress=ingress, generated_exclusive_writer_call=publication,
            expected_guard_for_existing_leaf='OUTPUT_ALREADY_OCCUPIED',
            runtime_error_path='EXISTING_DESTINATION_PRECHECK', exclusive_create_race_not_exercised=True),
        stable_input=dict(required_for_entire_process=True, predicate='CALLER_ATTESTS_FULL_INPUT_LIFETIME_STABILITY',
                         storage_locking_installed=False, mutation_scenario_exercised=False,
                         refusal='FULL_INPUT_LIFETIME_STABILITY_REQUIRED'),
        original_publication=dict(write_bytes=_source_anchor(source, b'.write_bytes('), fresh_scope_proven=True))


def _trace_command(tool, folder, command):
    if tool is None:
        return command
    return [tool['uri'], '-ff', '-qq', '-yy', '-s', '0', '-e',
            'trace=%file,write,writev', '-o', str(folder/'trace'), *command]


def _per_run_trace(folder, private, inputs):
    files = sorted(path for path in folder.iterdir() if re.fullmatch(r'trace\.[1-9][0-9]*', path.name))
    _need(files and len(files) <= 64, 'bounded per-thread syscall witness required')
    refs = [_ref(path) for path in files]
    stat_calls, opens, writes = [], [], []
    input_opened = []
    path_argument = json.dumps(str(private))
    descriptor = '<' + str(private) + '>'
    encoded_inputs = {json.dumps(row['path']): row['path'] for row in inputs}
    for ref in refs:
        for line in _raw(ref).decode('utf-8', errors='surrogateescape').splitlines():
            if line.startswith(('lstat(', 'newfstatat(', 'statx(')) and path_argument in line:
                stat_calls.append(line)
            if line.startswith(('open(', 'openat(', 'openat2(')):
                if path_argument in line and descriptor in line:
                    opens.append(line)
                input_opened.extend(path for argument, path in encoded_inputs.items() if argument in line)
            if line.startswith(('write(', 'writev(')) and descriptor in line:
                result = re.search(r'\)\s+= (\d+)\s*$', line)
                _need(result is not None, 'publication write did not return a known byte count')
                writes.append(int(result[1]))
    return dict(trace_files=refs, destination_stat_calls=stat_calls,
        destination_open_calls=opens, destination_write_counts=writes, input_files_opened=sorted(set(input_opened)))


def _capture(reference, package, private, commands, folder, mode, sentinel, tool):
    folder.mkdir()
    _need(not private.exists() and not private.is_symlink(), 'private destination unexpectedly occupied')
    with private.open('xb') as output:
        output.write(_raw(sentinel))
    private.chmod(_MODE)
    initial = _snapshot(private)
    (folder/'before.f32').write_bytes(_raw(sentinel))
    guards = _output_guards(private, package['publication']['expected_mode'])
    _need(guards == ['OUTPUT_ALREADY_OCCUPIED'], 'expected occupied-file preflight decline')
    declines = None
    if mode == _MAPPED:
        context = dict(cpu_ids=package['cpu_ids'], target=package['hardware'], environment=package['source_environment'])
        changed_argv = [*package['argv'][:-1], str(private)]
        changed = prepare_cpu_fixed_grid_native_route_launch(reference, argv=changed_argv,
            stable_inputs_for_process_lifetime=True, **context)
        unstable = prepare_cpu_fixed_grid_native_route_launch(reference, argv=package['argv'], **context)
        _need(changed['execution'] is None and changed['reasons'] == ['ARGV_DOES_NOT_MATCH']
              and unstable['execution'] is None and unstable['reasons'] == ['FULL_INPUT_LIFETIME_STABILITY_REQUIRED'],
              'the native package did not reject an out-of-scope request')
        declines = dict(changed_private_argv=changed, missing_input_stability=unstable)
    outcome = subprocess.run(_trace_command(tool, folder, commands[mode]), cwd=package['working_directory'],
        env=package['source_environment'], stdin=subprocess.DEVNULL, capture_output=True, timeout=package['timeout_seconds'])
    after = _snapshot(private)
    (folder/'stdout.bin').write_bytes(outcome.stdout)
    (folder/'stderr.bin').write_bytes(outcome.stderr)
    private.rename(folder/'after.f32')
    _write(folder/'outcome.json', dict(schema=_RUN, schema_version=1, before=initial, after=after,
        returncode=outcome.returncode, private_prelaunch_guard=guards, package_prelaunch_declines=declines))


def _replay_run(package, private, command, mode, folder, sentinel, tool):
    observation = _ref(folder/'outcome.json', _RUN)
    state = read_artifact(observation)
    before, after, stdout, stderr = [_ref(folder/name) for name in ('before.f32', 'after.f32', 'stdout.bin', 'stderr.bin')]
    pre, post = state['before'], state['after']
    final = package['publication']['expected_file']
    _need(_raw(before) == _raw(sentinel) and _signature(pre) == _signature(sentinel)
          and state['schema_version'] == 1 and state['private_prelaunch_guard'] == ['OUTPUT_ALREADY_OCCUPIED']
          and post == _snapshot(Path(after['uri']))
          and all(pre[key] == post[key] for key in ('device', 'inode', 'mode', 'link_count'))
          and pre['mode'] == _MODE, 'existing private inode, original bytes or retained file metadata differs')
    if mode == _ORIGINAL:
        _need(state['returncode'] == 0 and _raw(stdout) == _raw(package['stdout']) and _raw(stderr) == b''
              and _raw(after) == _raw(final) and _signature(post) != _signature(pre),
              'original did not truncate and replace the sentinel with exact final Tensor bytes')
    else:
        _need(state['returncode'] == 1 and _raw(stdout) == b'' and _raw(after) == _raw(before)
              and _raw(stderr) == f'SCAR ATen recurrence failed: output path already exists or is repeated: {private}\n'.encode(),
              'native occupied-output guard did not reject before modifying the sentinel')
    declines = state['package_prelaunch_declines']
    if mode == _MAPPED:
        for key, reason in (('changed_private_argv', 'ARGV_DOES_NOT_MATCH'),
                            ('missing_input_stability', 'FULL_INPUT_LIFETIME_STABILITY_REQUIRED')):
            _need(declines[key] == dict(status='DECLINE_NATIVE_ROUTE', reasons=[reason], execution=None,
                original_python_action='KEEP_ORIGINAL', fallback_after_execution=False,
                deployment_authorized=False, target_executed=False), 'native preflight receipt differs')
    else:
        _need(declines is None, 'original source trial has no native invocation authority')
    trace = None
    if tool is not None:
        trace = _per_run_trace(folder, private, package['file_inputs'])
        if mode == _ORIGINAL:
            _need(len(trace['destination_open_calls']) == 1 and 'O_TRUNC' in trace['destination_open_calls'][0]
                  and sum(trace['destination_write_counts']) == final['size_bytes'],
                  'original private file truncate/write was not observed')
        else:
            _need(any((line.startswith('lstat(') or 'AT_SYMLINK_NOFOLLOW' in line) and re.search(r'\)\s+= 0\s*$', line)
                      for line in trace['destination_stat_calls'])
                  and not any(trace[key] for key in ('destination_open_calls', 'destination_write_counts',
                                                    'input_files_opened')),
                  'native did not stop at the occupied-file query before intake and publication')
    return dict(mode=mode, command=command, diagnostic_command=_trace_command(tool, folder, command),
        returncode=state['returncode'], stdout=stdout, stderr=stderr, before=before, after=after,
        file_before=pre, file_after=post, private_prelaunch_guard=state['private_prelaunch_guard'],
        package_prelaunch_declines=declines, outcome=observation, trace=trace)


def _run(reference, root, tool, *, capture=False):
    package, commands, private, guards = _context(reference, root)
    stable = _input_files(package)
    if tool is not None:
        _need(tool == _ref(_TRACE, 'application/x-executable'), 'exact host syscall observer required')
    if capture:
        _need(set(package['cpu_ids']) <= os.sched_getaffinity(0), 'exact diagnostic CPU cohort is unavailable')
        root.mkdir(parents=True, mode=0o700, exist_ok=False)
        (root/'sentinel.f32').write_bytes(_SENTINEL)
    sentinel = _ref(root/'sentinel.f32')
    _need(_raw(sentinel) == _SENTINEL, 'private benign sentinel changed')
    runs = []
    for mode in _ORDER:
        folder = root/mode
        if capture:
            _capture(reference, package, private, commands, folder, mode, sentinel, tool)
        runs.append(_replay_run(package, private, commands[mode], mode, folder, sentinel, tool))
    original_output = Path(package['publication']['path'])
    _need(_input_files(package) == stable and all(not path.exists() and not path.is_symlink()
                                                for path in (private, original_output)),
          'inputs changed or a private/user output location was unexpectedly occupied')
    return dict(schema=SCHEMA, schema_version=1, status='VERIFIED_OUT_OF_DOMAIN_PUBLICATION_DIVERGENCE',
        route_package=reference, project=package['selected_project'], source=package['source'], target=package['hardware'],
        argv=package['argv'], cpu_ids=package['cpu_ids'], working_directory=package['working_directory'],
        environment_sha256=package['environment_sha256'], input_snapshots=stable, sentinel=sentinel, guards=guards,
        scope='EXISTING_PRIVATE_REGULAR_FILE_OUTSIDE_VERIFIED_FRESH_OUTPUT_DOMAIN', runs=runs, syscall_observer=tool,
        original_user_output_written=False, native_existing_file_preserved=True, original_existing_file_overwritten=True,
        original_default_retained=True, automatic_deployment=False, deployment_authorized=False, timing_measured=False)


def study(route_package_ref, out, *, trace=True):
    """Run two exact source CLIs once on a privately cloned existing file."""
    out = Path(out).absolute()
    _need(out.resolve() == out and not out.exists() and not out.is_symlink(), 'fresh canonical study directory required')
    tool = _ref(_TRACE, 'application/x-executable') if trace else None
    report = _run(route_package_ref, out, tool, capture=True)
    _write(out/'result.json', report)
    return report


def verify_cpu_fixed_grid_publication_guard_study(reference):
    """Replay source guard anchors, case outcomes and optional raw syscalls without target launches."""
    saved = read_artifact(reference)
    path = Path(reference['uri'])
    _need(reference['schema'] == SCHEMA and path.name == 'result.json' and path.resolve(strict=True) == path,
          'exact retained publication guard study required')
    _need(saved == _run(saved['route_package'], path.parent, saved['syscall_observer']),
          'retained publication guard observation changed at cold replay')
    return saved

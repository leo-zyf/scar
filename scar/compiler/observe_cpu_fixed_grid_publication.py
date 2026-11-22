"""Observe a successful fixed-grid publication at the Linux syscall boundary."""

from collections import Counter
import os
from pathlib import Path
import shutil
import stat
import subprocess

from scar.analysis.nvhpc_source_effects import _FD, _FOOTER, _OPEN_FLAGS, _QUOTED, _ROW, _WRITE, _child_fd, _path_argument
from scar.codegen.cpu_fixed_grid_native_route_package import verify_cpu_fixed_grid_native_route_package
from scar.compiler.calibrate_array_entry import _ref, _write
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.cpu-fixed-grid-publication-observation'
_LAUNCH_RESULT = 'scar.cpu-fixed-grid-effect-launch-result'
_TRACE = 'strace.syscalls.text.v1'
_RAW = 'application/octet-stream'
_ARMS = ('ORIGINAL_PYTHON', 'SELECTED_CPP')
_OPEN = {'open', 'openat', 'openat2', 'creat'}
_SYNC = {'fsync', 'fdatasync', 'sync_file_range', 'syncfs', 'sync', 'msync'}
_FORK = {'clone', 'clone3', 'fork', 'vfork'}
_NAMESPACE = {'chdir', 'fchdir', 'chroot', 'pivot_root', 'mkdir', 'mkdirat', 'rmdir', 'unlink', 'unlinkat',
    'rename', 'renameat', 'renameat2', 'link', 'linkat', 'symlink', 'symlinkat', 'mknod', 'mknodat',
    'truncate', 'ftruncate', 'chmod', 'fchmod', 'fchmodat', 'fchmodat2', 'chown', 'fchown', 'fchownat',
    'lchown', 'utime', 'utimes', 'utimensat', 'futimesat', 'setxattr', 'lsetxattr', 'fsetxattr',
    'removexattr', 'lremovexattr', 'fremovexattr'}
_READ_ONLY = {'stat', 'lstat', 'fstat', 'newfstatat', 'statx', 'statfs', 'fstatfs', 'access', 'faccessat',
    'faccessat2', 'readlink', 'readlinkat', 'getcwd', 'getxattr', 'lgetxattr', 'fgetxattr', 'listxattr',
    'llistxattr', 'flistxattr', 'name_to_handle_at', 'exit', 'exit_group', 'wait4', 'waitid'}
_FILTER = ','.join(('%file', '%network', '%process', 'close', 'mmap', 'fcntl', 'fchmod', 'fchown', 'ftruncate',
                    'fsetxattr', 'fremovexattr', 'fgetxattr', 'flistxattr', *sorted(_WRITE), *sorted(_SYNC)))


def _need(ok, detail):
    if not ok:
        raise ValueError('fixed-grid publication observation: ' + detail)


def _raw(reference):
    return read_artifact(reference, json_document=False)


def _file_metadata(path):
    value = path.lstat()
    _need(stat.S_ISREG(value.st_mode) and path.resolve(strict=True) == path, 'ordinary canonical result file required')
    return dict(mode=stat.S_IMODE(value.st_mode), uid=value.st_uid, gid=value.st_gid, links=value.st_nlink,
                device=value.st_dev, size_bytes=value.st_size)


def _capture(folder, command, package):
    from scar.codegen.cpu_fixed_grid_native_route_package import _output_guards

    path = Path(package['publication']['path'])
    _need(not _output_guards(path, package['publication']['expected_mode']), 'exact publication must be fresh and correctly scoped')
    folder.mkdir()
    run = subprocess.run(command, cwd=package['working_directory'], env=package['source_environment'],
                         stdin=subprocess.DEVNULL, capture_output=True, timeout=180)
    (folder/'stdout').write_bytes(run.stdout)
    (folder/'stderr').write_bytes(run.stderr)
    _need(run.returncode == 0 and path.exists(), 'diagnostic failed before successful publication')
    metadata = _file_metadata(path)
    shutil.move(path, folder/'published.f32')
    _write(folder/'execution.json', dict(schema=_LAUNCH_RESULT, returncode=run.returncode, publication=metadata))


def _effects(files, command, package, arm):
    output = package['publication']['path']
    names = {row['path'] for row in package['file_inputs']}
    images = set(package['launch_tools']) | {command[0]}
    if arm == 'ORIGINAL_PYTHON':
        images.add(command[3])
    else:
        images.add(package['routes'][package['selected_runtime']]['executable']['uri'])
    targets = {Path(row['uri']).suffix[1:] for row in files}
    read_inputs, process_ids, roots = set(), set(), []
    output_events, residual, extra_images, spawns, file_effects = [], Counter(), [], {}, []
    successful, failed, before_inputs = Counter(), Counter(), set()
    for index, trace in enumerate(files):
        task = Path(trace['uri']).suffix[1:]
        observed_input = False
        text = _raw(trace).decode('utf-8', errors='surrogateescape')
        for line_number, line in enumerate(text.splitlines(), 1):
            where = dict(trace=index, line=line_number)
            if (exit_event := _FOOTER.fullmatch(line)) is not None:
                if exit_event[1] != '0':
                    residual['NONZERO_TASK_EXIT'] += 1
                continue
            matched = _ROW.fullmatch(line)
            if matched is None:
                if not line.startswith('--- '):
                    residual['UNPARSED_TRACE_ROW'] += 1
                continue
            call, args, result = matched.groups()
            if result.startswith('-1 '):
                failed[call] += 1
                continue
            successful[call] += 1
            window = ('FOLLOWED_TASK' if index not in roots else
                      'OWNER_AFTER_FIRST_INPUT' if observed_input else 'OWNER_BEFORE_FIRST_INPUT')
            if call in ('execve', 'execveat'):
                image = _path_argument(args)
                if image == command[0]:
                    roots.append(index)
                if image not in images:
                    residual['OTHER_EXEC_IMAGE'] += 1
                    extra_images.append(dict(image=image, task=task, **where))
            elif call in _FORK:
                if result.isdecimal():
                    process_ids.add(result)
                if call in ('fork', 'vfork') or 'CLONE_THREAD' not in args:
                    residual['PROCESS_SPAWN'] += 1
                    if result.isdecimal():
                        spawns[result] = dict(parent_task=task, before_first_input=not observed_input, syscall=call, **where)
            elif call in _OPEN:
                descriptor = _FD.fullmatch(result)
                path = descriptor[2] if descriptor is not None else None
                quote = _QUOTED.search(args)
                flags = args[quote.end():] if quote else args
                open_flags = sorted(set(part for part in flags.replace('|', ' ').replace(',', ' ').split()
                                        if part.startswith('O_')))
                if path == output:
                    output_events.append(dict(kind='OPEN', flags=open_flags, **where))
                elif call == 'creat' or _OPEN_FLAGS.search(flags):
                    residual['OTHER_WRITE_OPEN'] += 1
                    file_effects.append(dict(kind='WRITE_CAPABLE_OPEN', path=_path_argument(args),
                                             descriptor_target=path, flags=open_flags, window=window, syscall=call, **where))
                elif path in names:
                    observed_input = True
                    read_inputs.add(path)
                elif path and path.startswith('/') and not observed_input and index in roots:
                    before_inputs.add(path)
            elif call in _WRITE or call == 'close':
                fd, path = _child_fd(args)
                if path == output:
                    output_events.append(dict(kind='CLOSE' if call == 'close' else 'WRITE', bytes=int(result), **where))
                elif call in _WRITE and fd == 1 and path and path.startswith('pipe:[') and index in roots:
                    output_events.append(dict(kind='STDOUT', bytes=int(result), **where))
                elif call in _WRITE:
                    residual['OTHER_WRITE'] += 1
            elif call in _SYNC:
                residual['SYNC_INTERFACE'] += 1
            elif call in _NAMESPACE:
                residual['NAMESPACE_OR_METADATA_MUTATION'] += 1
                file_effects.append(dict(kind='NAMESPACE_OR_METADATA_MUTATION', path=_path_argument(args),
                                         window=window, syscall=call, **where))
            elif call == 'mmap':
                if 'MAP_SHARED' in args and 'PROT_WRITE' in args and 'MAP_ANONYMOUS' not in args:
                    residual['WRITABLE_SHARED_MAPPING'] += 1
            elif call == 'fcntl' and any(name in args for name in ('F_GETFD', 'F_GETFL', 'F_SETFD')):
                continue
            elif call not in _READ_ONLY:
                residual['OTHER_CAPTURED_INTERFACE'] += 1

    expected_policy = 'O_TRUNC' if arm == 'ORIGINAL_PYTHON' else 'O_EXCL'
    kinds = [row['kind'] for row in output_events]
    exact_order = (len(set(roots)) == 1 and {row['trace'] for row in output_events} == set(roots)
                   and len(kinds) >= 4 and kinds[0] == 'OPEN' and 'CLOSE' in kinds
                   and kinds[1:kinds.index('CLOSE')] == ['WRITE'] * (kinds.index('CLOSE')-1)
                   and all(kind == 'STDOUT' for kind in kinds[kinds.index('CLOSE') + 1:])
                   and kinds[-1] == 'STDOUT')
    _need(exact_order and expected_policy in output_events[0]['flags'] and 'O_CREAT' in output_events[0]['flags']
          and sum(row['bytes'] for row in output_events if row['kind'] == 'WRITE') == package['publication']['expected_file']['size_bytes']
          and sum(row['bytes'] for row in output_events if row['kind'] == 'STDOUT') == package['stdout']['size_bytes'],
          'normal file creation, exact write and close-before-stdout must be observed in the owner trace')
    _need(read_inputs == names, 'one or more admitted input paths were not observed')
    if not process_ids <= targets:
        residual['UNMATCHED_TRACED_CHILDREN'] = len(process_ids-targets)
    created = {(row['trace'], row['path']): row for row in file_effects
               if row['kind'] == 'WRITE_CAPABLE_OPEN' and {'O_CREAT', 'O_EXCL'} <= set(row['flags'])}
    transient = [dict(path=row['path'], create=created[row['trace'], row['path']]['line'], unlink=row['line'],
                      trace=row['trace'], window=row['window']) for row in file_effects
                 if row['syscall'] == 'unlink' and (row['trace'], row['path']) in created
                 and created[row['trace'], row['path']]['line'] < row['line']]
    return dict(events=output_events, inputs_opened_read_only=sorted(read_inputs), syscall_counts=dict(successful),
                failed_syscall_counts=dict(failed), other_effects=dict(residual),
                external_exec_images=[dict(row, spawn=spawns.get(row['task'])) for row in extra_images],
                other_file_effects=file_effects, observed_same_task_create_unlink=transient,
                distinct_owner_read_paths_before_first_input=len(before_inputs),
                normal_close_before_stdout_observed=True)


def _arm(folder, arm, original, package, tool, *, capture):
    route = package['routes'][package['selected_runtime']]
    command = original if arm == 'ORIGINAL_PYTHON' else [route['launcher']['uri'], *package['argv']]
    traced = [tool['uri'], '-ff', '-yy', '-s', '0', '--kill-on-exit', '-e', 'trace='+_FILTER,
              '-o', str(folder/'effect'), '--', *command]
    if capture:
        _capture(folder, traced, package)
    files = sorted(folder.glob('effect.*'))
    _need(1 <= len(files) <= 64 and sum(path.stat().st_size for path in files) <= 8 * 1024**2
          and all(path.suffix[1:].isdigit() and path.resolve(strict=True) == path for path in files),
          'bounded ordinary per-task trace files required')
    traces = [_ref(path, _TRACE) for path in files]
    launch = _ref(folder/'execution.json', _LAUNCH_RESULT)
    raw = read_artifact(launch)
    stdout, stderr, file = (_ref(folder/name, _RAW) for name in ('stdout', 'stderr', 'published.f32'))
    expected = package['publication']['expected_file']
    _need(raw['returncode'] == 0 and raw['publication'] == _file_metadata(folder/'published.f32')
          and raw['publication']['mode'] == package['publication']['expected_mode']
          and _raw(stdout) == _raw(package['stdout']) and _raw(stderr) == b''
          and all(file[key] == expected[key] for key in ('sha256', 'size_bytes')),
          'traced arm changed expected streams, successful exit, output metadata or bytes')
    return dict(command=command, diagnostic_command=traced, traces=traces, launch=launch, stdout=stdout,
                stderr=stderr, file=file, publication=raw['publication'], trace_projection=_effects(traces, command, package, arm))


def _record(package_ref, root, tool, *, capture=False):
    package = verify_cpu_fixed_grid_native_route_package(package_ref)
    delivery = read_artifact(package['owned_copy_delivery'])
    original = delivery['launch']['commands']['ORIGINAL_PYTHON']
    _need(original[0] == '/usr/bin/taskset' and original[1] == '-c' and original[5:] == [package['source']['uri'], *package['argv']],
          'exact direct original Python CLI required')
    for source in (tool, *[dict(uri=row['path'], schema=_RAW, sha256=row['sha256'], size_bytes=row['size_bytes'])
                          for row in package['file_inputs']]):
        _raw(source)
    if capture:
        _need(set(package['cpu_ids']) <= os.sched_getaffinity(0), 'recorded source CPU cohort unavailable')
        root.mkdir(parents=True, exist_ok=False)
    arms = {arm: _arm(root/arm, arm, original, package, tool, capture=capture) for arm in _ARMS}
    for row in package['file_inputs']:
        _raw(dict(uri=row['path'], schema=_RAW, sha256=row['sha256'], size_bytes=row['size_bytes']))
    return dict(schema=SCHEMA, schema_version=1, package=package_ref, delivery=package['owned_copy_delivery'],
        source=package['source'], target=package['hardware'], argv=package['argv'], cpu_ids=package['cpu_ids'],
        environment_sha256=package['environment_sha256'], strace=tool, observed_interfaces=_FILTER, arms=arms,
        matching_file_metadata=arms[_ARMS[0]]['publication'] == arms[_ARMS[1]]['publication'],
        return_content_and_streams_verified=True, diagnostic_only=True, persistent_storage_durability_proven=False,
        concurrent_observer_equivalence_proven=False, unobserved_effect_surfaces=['IOCTL', 'SIGNALS', 'ASYNC_IO', 'SHARED_MEMORY_STORES'],
        whole_entry_effect_equivalence_proven=False, deployment_authorized=False)


def measure_cpu_fixed_grid_publication(package_ref, out, *, strace='/usr/bin/strace'):
    root = Path(out).absolute()
    _need(root.resolve() == root, 'canonical diagnostic directory required')
    tool = _ref(Path(strace).resolve(strict=True), 'application/x-executable')
    report = _record(package_ref, root, tool, capture=True)
    _write(root/'result.json', report)
    return report


def verify_cpu_fixed_grid_publication(reference):
    saved = read_artifact(reference)
    path = Path(reference['uri'])
    _need(reference['schema'] == SCHEMA and path.name == 'result.json' and saved == _record(
        saved['package'], path.parent, saved['strace']), 'publication diagnostic differs at cold replay')
    return saved

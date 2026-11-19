"""Collect bounded aggregates from an independently rebuilt ordered CPU solver."""

import argparse
import json
import os
from pathlib import Path
import re
from shutil import copytree
import struct
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from scar.codegen import native_ordered_profile as profile
from scar.codegen import native_ordered_sparse_export as ordered
from scar.codegen.aten_provider import _dependencies
from scar.codegen.native_project import build_native_project
from scar.codegen.native_provider_bundle import _content
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.native_ordered_publication import ENV


SCHEMA = 'scar.native-ordered-timing-observation'
_SOURCE = 'scar.cpp-source'
_HEADER = 'scar.cpp-header'
_EXECUTABLE = 'scar.native-executable-file'
_STDOUT = 'application/octet-stream'
_INPUT = 'scar.native-ordered-source-csr-bytes'
_TIMING = 'scar.native-ordered-timing-aggregates-tsv'
_TEMPLATE = Path(__file__).with_name('native_ordered_timing_v1.hpp.in')
_PROBE = 'native_ordered_timing_v1.hpp'
_TIMING_FD = 'SCAR_ORDERED_TIMING_FD'
_OWNER_SINGLE = ('entry', 'ingress', 'dispatch_bootstrap', 'publication_finalize', 'drain_reap')
_OWNER_ITEMS = ('issue_prepare', 'issue_send', 'ordered_wait_receive_header', 'ordered_receive_payload',
                'retire', 'digest_update')
_WORKER_SINGLE = ('lifetime', 'bootstrap', 'close_receive')
_WORKER_ITEMS = ('receive_request', 'solve', 'numeric_provider_solve', 'send_response')
_FIELDS = ('count', 'total_wall_ns', 'total_cpu_ns', 'min_wall_ns', 'max_wall_ns', 'min_cpu_ns', 'max_cpu_ns')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _member(root, reference):
    return {**reference, 'uri': str(root / reference['uri'])}


def _package(path):
    path = Path(path).resolve(strict=True)
    root = path if path.is_dir() else path.parent
    manifest = root / 'artifact.json'
    _require(path == root or path == manifest, 'ordered diagnostic requires the production package or manifest')
    schema = json.loads(manifest.read_bytes())['schema']
    if schema == profile.SCHEMA:
        package = profile.check_profile_solver(root)
        resources = read_artifact(package['profile_plan'])['resources']
    elif schema == ordered.SCHEMA:
        package = ordered.check_native_ordered_export(root)
        resources = read_artifact(_member(root, package['plan']))
    else:
        raise ValueError('unsupported production solver container for ordered timings')
    stream = read_artifact(_member(root, package['stream']['protocol']))
    transport = package['stream']
    placement = {key: resources['placement'][key] for key in (
        'owner_cpu_id', 'worker_cpu_ids', 'launch_cpu_ids', 'cpu_numa_node')}
    count = resources['case']['call_count']
    width = len(placement['worker_cpu_ids'])
    _require(stream['schema'] == ordered.DIGEST_STREAM and transport['source']['uri'] == ordered._STREAM_MAIN
             and transport['cmake']['uri'] == 'CMakeLists.txt'
             and transport['executable']['uri'] == ordered._STREAM_EXECUTABLE
             and count == stream['header']['count'] and width == resources['dispatch']['window']
             and 1 <= width <= count, 'ordered diagnostic requires a complete canonical CSR-to-digest cohort')
    context = dict(package=_ref(manifest, schema), recipe_id=package['recipe_id'], placement=placement,
                   count=count, width=width, protocol=_member(root, transport['protocol']))
    return root, package, stream, context


def _input(reference, protocol):
    raw = read_artifact(reference, json_document=False)
    header = protocol['header']
    fields = struct.Struct(header['format'])
    expected = (protocol['input']['magic'].encode('ascii'), bytes.fromhex(header['recipe_digest_hex']),
                header['size'], header['nnz'], header['count'], header['flags'])
    fixed = protocol['bytes']['input_fixed']
    _require(fixed <= len(raw) <= protocol['bytes']['input_complete_max'] and fields.unpack_from(raw) == expected,
             'canonical CSR input differs from the exact ordered recipe header')
    rows, = struct.unpack_from('<I', raw, fixed - 4)
    _require(rows <= header['size'] and len(raw) == fixed + rows * protocol['bytes']['selected_row'],
             'canonical CSR selected-row extent differs from the ordered input protocol')


def _patch(source, undiagnosed, diagnosed):
    _require(source.count(undiagnosed) == 1, 'ordered timing probe source location is absent or ambiguous')
    return source.replace(undiagnosed, diagnosed, 1)


def _span(value, phase):
    return f'probe::measure(probe::Phase::{phase}, [&] {{ return {value}; }})'


def _render_dispatch(raw):
    source = _patch(raw.decode(), 'namespace scar::ordered_v1 {',
        f'#include "{_PROBE}"\n\nnamespace scar::ordered_v1 {{\n\nnamespace probe = scar::ordered_timing_v1;')
    patches = (
        ('    ~Team() {', '    ~Team() {\n        probe::Scope drain(probe::Phase::drain_reap);'),
        ('    const auto cap = static_cast<rlim_t>', '    probe::Scope lifetime(probe::Phase::lifetime);\n'
         '    probe::Scope bootstrap(probe::Phase::bootstrap);\n    const auto cap = static_cast<rlim_t>'),
        ('    try {\n        std::vector<std::byte> request', '    probe::cpu_start();\n    try {\n'
         '        std::vector<std::byte> request'),
        ('        std::uint64_t sequence;\n        while (transfer(fd, &sequence, sizeof(sequence), false)) {',
         '        bootstrap.finish();\n        std::uint64_t sequence;\n        while (true) {\n'
         '            probe::Scope receive(probe::Phase::receive_request);\n'
         '            if (!transfer(fd, &sequence, sizeof(sequence), false)) {\n'
         '                receive.finish_as(probe::Phase::close_receive);\n                break;\n            }'),
        ('            const auto status = worker(sequence, std::span<const std::byte>(request), std::span<std::byte>(output));',
         '            receive.finish();\n            const auto status = ' + _span(
          'worker(sequence, std::span<const std::byte>(request), std::span<std::byte>(output))', 'solve') + ';'),
        ('            Response response{sequence, status, 0};', '            Response response{sequence, status, 0};\n'
         '            probe::Scope send(probe::Phase::send_response);'),
        ('        return 0;\n    } catch (...) {', '        probe::cpu_end();\n        return 0;\n    } catch (...) {'),
        ('    detail::require_config(selected);', '    probe::Scope bootstrap(probe::Phase::dispatch_bootstrap);\n'
         '    detail::require_config(selected);'),
        ('    detail::OwnerAffinity affinity(config);', '    detail::OwnerAffinity affinity(config);\n    probe::cpu_start();'),
        ('        if (pid == 0) {\n            close(pair[0]);', '        if (pid == 0) {\n'
         '            probe::worker_slot(i);\n            close(pair[0]);'),
        ('\n    const auto issue =', '\n    bootstrap.finish();\n    const auto issue ='),
        ('        fill(sequence, std::span<std::byte>(request));', '        ' +
         _span('fill(sequence, std::span<std::byte>(request))', 'issue_prepare') + ';\n'
         '        probe::Scope send(probe::Phase::issue_send);'),
        ('        slot.outstanding = true;', '        send.finish();\n        slot.outstanding = true;'),
        ('detail::transfer(slot.fd, &response, sizeof(response), false, until)',
         _span('detail::transfer(slot.fd, &response, sizeof(response), false, until)',
               'ordered_wait_receive_header')),
        ('detail::transfer(slot.fd, output.data(), output.size(), false, until)',
         _span('detail::transfer(slot.fd, output.data(), output.size(), false, until)', 'ordered_receive_payload')),
        ('        retire(result);', '        ' + _span('retire(result)', 'retire') + ';'),
        ('    return summary;\n}', '    probe::cpu_end();\n    return summary;\n}'),
    )
    for before, after in patches:
        source = _patch(source, before, after)
    return source.encode()


def _render_entry(raw):
    source = f'#include "{_PROBE}"\n\nnamespace probe = scar::ordered_timing_v1;\n\n' + raw.decode()
    patches = (
        ('        auto observed = input_header;', '        probe::Probe metrics(std::size(sparse::worker_cpus));\n'
         '        probe::Scope entry(probe::Phase::entry);\n        probe::Scope ingress(probe::Phase::ingress);\n'
         '        auto observed = input_header;'),
        ('        return run_requests(indices, offsets, input);', '        ingress.finish();\n'
         '        const auto status = run_requests(indices, offsets, input);\n        entry.finish();\n'
         '        if (status == 0) {\n            metrics.publish();\n        }\n        return status;'),
        ('                const auto status = provider->solve(symbolic.get(),',
         '                probe::Scope numeric(probe::Phase::numeric_provider_solve);\n'
         '                const auto status = provider->solve(symbolic.get(),'),
        ('rhs.data(), solution.data(), &timing);\n                if (status == 0) {',
         'rhs.data(), solution.data(), &timing);\n                numeric.finish();\n                if (status == 0) {'),
        ('            publication.append(row.result);', '            ' + _span('publication.append(row.result)', 'digest_update') + ';'),
        ('        publication.complete();\n        return 0;', '        ' + _span('publication.complete()', 'publication_finalize') +
         ';\n        return 0;'),
    )
    for before, after in patches:
        source = _patch(source, before, after)
    return source.encode()


def _render(root, package):
    headers = {row['uri']: read_artifact(_member(root, row), json_document=False) for row in package['headers']}
    headers[ordered._DISPATCH_HEADER] = _render_dispatch(headers[ordered._DISPATCH_HEADER])
    headers[_PROBE] = _TEMPLATE.read_bytes()
    sources = {name: (raw, _HEADER) for name, raw in headers.items()}
    sources[ordered._STREAM_MAIN] = (_render_entry(read_artifact(_member(root, package['stream']['source']),
                                                                json_document=False)), _SOURCE)
    sources['CMakeLists.txt'] = (read_artifact(_member(root, package['stream']['cmake']), json_document=False),
                                 'scar.cmake-project')
    return sources


def _libraries(source, diagnostic):
    def tree(root):
        return {str(path.relative_to(root)): _content(path) for name in ('numeric', 'sha256')
                for path in (root / name).rglob('*') if path.is_file()}

    before, after = tree(source), tree(diagnostic)
    _require(before and before == after, 'diagnostic numerical or digest provider bytes differ from production')
    return before


def _linkage(path):
    return _dependencies(subprocess.check_output(['/usr/bin/ldd', str(path)], env={}, text=True))


def _output(path):
    value = path.read_bytes()
    _require(re.fullmatch(rb'[0-9a-f]{64}\n', value) is not None, 'ordered solver did not publish one SHA-256 line')
    return value


def _read_timing(path, context):
    header, *records = path.read_text(encoding='ascii').splitlines()
    _require(header == 'SCAR_ORDERED_TIMING_V1', 'invalid native timing records')
    lanes = []
    for row in records:
        key, *raw = row.split()
        values = list(map(int, raw))
        if key == 'lane':
            i, start, end = values
            _require(i == len(lanes), 'out-of-order diagnostic lanes')
            lanes.append(dict(slot=i, cpu_start=start, cpu_end=end, phases={}))
        else:
            phases = lanes[-1]['phases']
            _require(key not in phases and len(values) == len(_FIELDS), 'duplicate or malformed diagnostic phase')
            phases[key] = dict(zip(_FIELDS, values, strict=True))
    placement = context['placement']
    cpus = [placement['owner_cpu_id'], *placement['worker_cpu_ids']]
    count, width = context['count'], context['width']
    _require(len(lanes) == len(cpus), 'diagnostic lane count differs from the exact cohort')
    for slot, (lane, cpu) in enumerate(zip(lanes, cpus, strict=True)):
        _require(lane['cpu_start'] == lane['cpu_end'] == cpu, 'diagnostic CPU differs from the selected pin')
        single, items = (_OWNER_SINGLE, _OWNER_ITEMS) if slot == 0 else (_WORKER_SINGLE, _WORKER_ITEMS)
        n = count if slot == 0 else (count + width - slot) // width
        expected = {**dict.fromkeys(single, 1), **dict.fromkeys(items, n)}
        _require(lane['phases'].keys() == expected.keys(), 'missing or unexpected diagnostic timing phases')
        for name, times in lane['phases'].items():
            samples = times['count']
            _require(samples == expected[name], 'diagnostic phase count differs from issued sequence scope')
            for clock in ('wall', 'cpu'):
                low, high, total = (times[f'{bound}_{clock}_ns'] for bound in ('min', 'max', 'total'))
                _require(0 <= low <= high and (samples - 1) * low + high <= total <= (samples - 1) * high + low,
                         'inconsistent diagnostic clock aggregate')
    return dict(owner=lanes[0], workers=lanes[1:], clocks=['CLOCK_MONOTONIC', 'CLOCK_PROCESS_CPUTIME_ID'],
                marker_scope='NESTED_PHASES_DO_NOT_ADD', receive_scope='WAIT_WAKE_AND_RECV_WALL_NOT_PURE_IPC',
                cost_scope='INSTRUMENTED_NATIVE_DIAGNOSTIC_NOT_PROFITABILITY_EVIDENCE')


def _run(executable, input_path, cpus, *, timing=None):
    command = ['/usr/bin/taskset', '-c', ','.join(map(str, cpus)), str(executable)]
    environment = dict(ENV)
    descriptors = ()
    if timing is not None:
        fd = timing.fileno()
        environment[_TIMING_FD] = str(fd)
        descriptors = (fd,)
    with input_path.open('rb') as stream:
        started = perf_counter_ns()
        run = subprocess.run(command, cwd='/', env=environment, stdin=stream, capture_output=True,
                             timeout=125, pass_fds=descriptors)
        elapsed = perf_counter_ns() - started
    _require(run.returncode == 0 and not run.stderr, 'ordered native diagnostic launch failed: ' +
             run.stderr.decode(errors='replace').strip())
    return run.stdout, elapsed


def collect(package, out, *, input_path, expected_stdout=None):
    """Run one exact production/diagnostic stdout check, with aggregate diagnostic times."""
    root, manifest, protocol, context = _package(package)
    input_ref = _ref(input_path, _INPUT)
    _input(input_ref, protocol)
    oracle = _ref(expected_stdout, _STDOUT) if expected_stdout is not None else None
    expected = _output(Path(oracle['uri'])) if oracle is not None else None
    out = Path(out).resolve()
    _require(not out.exists() and not out.is_relative_to(root), 'a new diagnostic directory outside production is required')
    out.parent.mkdir(parents=True, exist_ok=True)
    compiler = manifest['stream']['compiler']
    current = subprocess.check_output([*compiler['command'], '--version'], text=True).splitlines()[0]
    _require(compiler['version'] == current, 'ordered diagnostic compiler version differs from production')
    with TemporaryDirectory(prefix='.scar-ordered-timing-', dir=out.parent) as temporary:
        stage = Path(temporary)
        sources = _render(root, manifest)
        for name, (raw, _) in sources.items():
            (stage / name).write_bytes(raw)
        for name in ('numeric', 'sha256'):
            copytree(root / name, stage / name)
        built, _ = build_native_project(stage, ordered._STREAM_EXECUTABLE, compiler['command'])
        executable = built.rename(stage / ordered._STREAM_EXECUTABLE)
        production = root / ordered._STREAM_EXECUTABLE
        _require(_linkage(production) == _linkage(executable), 'direct diagnostic ELF linkage differs from production')
        cpus = context['placement']['launch_cpu_ids']
        first, clean_time = _run(production, Path(input_ref['uri']), cpus)
        with (stage / 'timing.tsv').open('xb') as stream:
            second, diag_time = _run(executable, Path(input_ref['uri']), cpus, timing=stream)
        _require(first == second and (expected is None or first == expected), 'native diagnostic publication differs')
        _input(input_ref, protocol)
        for label, data in (('production', first), ('diagnostic', second)):
            (stage / f'{label}.stdout').write_bytes(data)
            _output(stage / f'{label}.stdout')
        record = dict(schema=SCHEMA, schema_version=1, **context, input=input_ref, expected_stdout=oracle,
                      production=dict(executable=_member(root, manifest['stream']['executable']),
                                      stdout={**_ref(stage / 'production.stdout', _STDOUT), 'uri': 'production.stdout'},
                                      wall_ns=clean_time),
                      diagnostic=dict(executable=ordered._local(stage, executable.name, _EXECUTABLE),
                                      stdout=ordered._local(stage, 'diagnostic.stdout', _STDOUT), wall_ns=diag_time),
                      compiler=compiler, libraries=_libraries(root, stage), linkage=_linkage(executable),
                      sources=[ordered._local(stage, name, kind) for name, (_, kind) in sources.items()],
                      evidence=ordered._local(stage, 'timing.tsv', _TIMING), timing=_read_timing(stage / 'timing.tsv', context),
                      runtime=dict(cwd='/', environment=ENV, diagnostic_addition='PRIVATE_INHERITED_FILE_DESCRIPTOR'),
                      deployment_authorized=False)
        (stage / 'result.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
        if out.exists():
            raise FileExistsError(out)
        stage.rename(out)
    return _ref(out / 'result.json', SCHEMA)


def check(path):
    """Replay exact production provenance, diagnosis source and recorded phase invariants."""
    ref = _ref(path, SCHEMA)
    record = read_artifact(ref)
    result = Path(ref['uri']).parent
    root, manifest, protocol, context = _package(record['package']['uri'])
    _require(record['schema_version'] == 1 and all(record.get(key) == value for key, value in context.items())
             and record['deployment_authorized'] is False and record['input']['schema'] == _INPUT,
             'ordered diagnostic receipt differs from its production recipe')
    _input(record['input'], protocol)
    sources = _render(root, manifest)
    _require(record['sources'] == [ordered._local(result, name, kind) for name, (_, kind) in sources.items()]
             and all((result / name).read_bytes() == raw for name, (raw, _) in sources.items()),
             'ordered diagnostic source edits differ from the deterministic probe placement')
    executable = result / ordered._STREAM_EXECUTABLE
    clean = _member(root, manifest['stream']['executable'])
    _require(record['production']['executable'] == clean and os.access(executable, os.X_OK)
             and record['diagnostic']['executable'] == ordered._local(result, executable.name, _EXECUTABLE)
             and record['compiler'] == manifest['stream']['compiler']
             and record['libraries'] == _libraries(root, result)
             and record['linkage'] == _linkage(Path(clean['uri'])) == _linkage(executable)
             and record['runtime'] == dict(cwd='/', environment=ENV, diagnostic_addition='PRIVATE_INHERITED_FILE_DESCRIPTOR'),
             'ordered diagnostic compiler, linkage or runtime context differs')
    clean_output = _output(result / 'production.stdout')
    for label in ('production', 'diagnostic'):
        run = record[label]
        path = result / f'{label}.stdout'
        _require(type(run['wall_ns']) is int and run['wall_ns'] > 0 and _output(path) == clean_output
                 and run['stdout'] == ordered._local(result, path.name, _STDOUT), 'ordered diagnostic publication differs')
    if record['expected_stdout'] is not None:
        _require(read_artifact(record['expected_stdout'], json_document=False) == clean_output,
                 'ordered diagnostic differs from the original output oracle')
    timing = result / 'timing.tsv'
    _require(record['evidence'] == ordered._local(result, timing.name, _TIMING)
             and record['timing'] == _read_timing(timing, context), 'ordered diagnostic timing evidence differs')
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package')
    parser.add_argument('--input')
    parser.add_argument('--expected-stdout')
    parser.add_argument('--out')
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check and not any((args.package, args.input, args.expected_stdout, args.out)):
        result = check(args.check)
    elif args.package and args.input and args.out and not args.check:
        result = collect(args.package, args.out, input_path=args.input, expected_stdout=args.expected_stdout)
    else:
        parser.error('provide --package, --input and --out, or --check RESULT')
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()

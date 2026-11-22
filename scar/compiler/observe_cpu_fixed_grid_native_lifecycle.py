"""Observe exact native fixed-grid lifecycle cuts in a separate C++20 build."""

from hashlib import sha256
import os
from pathlib import Path
import shutil
import stat
from struct import Struct
import subprocess
from time import CLOCK_MONOTONIC, clock_gettime_ns

from scar.codegen.native_cpu_elf import _dependencies
from scar.codegen.aten_recurrence_source import FRESH_SOURCE_FILE
from scar.codegen.native_cpu_elf import cpu_executable_dynamic
from scar.codegen.native_project import build_native_project
from scar.codegen.native_provider_bundle import _content
from scar.compiler.aten_source_entry import LAUNCHER
from scar.compiler.calibrate_array_entry import _write
from scar.compiler.cpu_fixed_grid_cli_delivery import verify_cpu_fixed_grid_cli_delivery
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.cpu-fixed-grid-native-lifecycle'
_CLOCK_SCHEMA = 'scar.cpu-fixed-grid-native-lifecycle-clock'
_MAIN = 'src/main.cpp'
_FILE = 'src/runtime/file_io.hpp'
_HEADER = 'src/runtime/scar_fixed_grid_owner_clock.hpp'
_SUFFIX = '.scar-native-clock.bin'
_RAW = 'application/octet-stream'
MARKERS = ('MAIN_ENTER', 'RUNTIME_BEGIN', 'TEAM_CONSTRUCTOR_RETURN', 'INPUTS_BEGIN', 'INPUTS_READY',
           'RECURRENCE_BEGIN', 'RECURRENCE_RETURN', 'FILE_PUBLICATION_BEGIN', 'FILE_PUBLICATION_RETURN',
           'STDOUT_BEGIN', 'STDOUT_SUBMIT_END', 'INVOKE_RETURN', 'MAIN_END')
WINDOWS = dict(launcher_to_main=('launcher_start_ns', 'MAIN_ENTER'), main_ingress=('MAIN_ENTER', 'RUNTIME_BEGIN'),
    runtime_team_setup=('RUNTIME_BEGIN', 'TEAM_CONSTRUCTOR_RETURN'),
    input_preflight=('TEAM_CONSTRUCTOR_RETURN', 'INPUTS_BEGIN'), input_ingestion=('INPUTS_BEGIN', 'INPUTS_READY'),
    recurrence=('RECURRENCE_BEGIN', 'RECURRENCE_RETURN'),
    file_publication=('FILE_PUBLICATION_BEGIN', 'FILE_PUBLICATION_RETURN'),
    stdout_submission=('STDOUT_BEGIN', 'STDOUT_SUBMIT_END'),
    owner_local_teardown=('INVOKE_RETURN', 'MAIN_END'), main_end_to_reap=('MAIN_END', 'process_reaped_ns'))
_CLOCKS = Struct('=' + 'Q' * len(MARKERS))
USAGE_MARKERS = ('INPUTS_BEGIN', 'INPUTS_READY', 'RECURRENCE_BEGIN', 'RECURRENCE_RETURN')
USAGE_FIELDS = ('minor_faults', 'major_faults', 'block_input_events')
_USAGE = Struct('=4Q')


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU fixed-grid native lifecycle: ' + detail)


def _raw(reference):
    return read_artifact(reference, json_document=False)


def _mode(path):
    mode = path.lstat().st_mode
    _need(stat.S_ISREG(mode) and path.resolve(strict=True) == path, 'exact ordinary artifact file required: ' + str(path))
    return stat.S_IMODE(mode)


def _ref(path, schema=_RAW):
    _mode(path)
    return artifact_reference(path, schema)


def _clock_header():
    markers = ', '.join(MARKERS)
    return f'''#pragma once
#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <time.h>

namespace scar::fixed_grid_clock {{
enum Marker {{ {markers}, COUNT }};
static std::array<std::uint64_t, COUNT> stamps{{}};
static_assert(sizeof(stamps) == {8 * len(MARKERS)});

inline void stamp(Marker point) {{
    timespec observed{{}};
    if (clock_gettime(CLOCK_MONOTONIC, &observed) == 0) {{
        stamps[point] = std::uint64_t(observed.tv_sec) * 1000000000ULL + std::uint64_t(observed.tv_nsec);
    }}
}}

struct MainWitness {{
    const char* destination = nullptr;
    ~MainWitness() {{
        if (!destination) return;
        stamp(MAIN_END);
        const auto sidecar = std::string(destination) + "{_SUFFIX}";
        auto* out = std::fopen(sidecar.c_str(), "wbx");
        if (!out) return;
        std::fwrite(stamps.data(), sizeof(stamps), 1, out);
        std::fclose(out);
    }}
}};
}}
'''.encode('ascii')


class _SourceEdits:
    def __init__(self, reference, source):
        self.reference = reference
        self.source = source
        self.edits = []
        self.sites = {}

    def mark(self, marker, statement, *, after=False, source_only=False):
        padding = statement[:len(statement) - len(statement.lstrip(b' '))]
        guard = b'if (fresh_source_file) ' if source_only else b''
        point = f'scar::fixed_grid_clock::stamp(scar::fixed_grid_clock::{marker});\n'.encode('ascii')
        self.insert(marker, statement, padding + guard + point, after=after)

    def _site(self, marker, anchor, position):
        _need(self.source.count(anchor) == 1 and marker not in self.sites, 'expected one owner source anchor for ' + marker)
        begin = self.source.index(anchor)
        end = begin + len(anchor)
        offset = end if position == 'AFTER' else begin
        _need(offset > 0 and self.source[offset-1:offset] == b'\n', 'mark does not lie on an exact statement boundary')
        self.sites[marker] = dict(source=self.reference, byte_offset=offset, placement=position,
                                  anchor=dict(byte_start=begin, byte_end=end, sha256=sha256(anchor).hexdigest()))
        return offset

    def insert(self, marker, anchor, statement, *, after=False):
        offset = self._site(marker, anchor, 'AFTER' if after else 'BEFORE')
        self.edits.append((offset, offset, statement))

    def rewrite(self, marker, anchor, value):
        offset = self._site(marker, anchor, 'REPLACE')
        self.edits.append((offset, offset + len(anchor), value))

    def render(self, prefix=b''):
        chunks = [prefix]
        begin = 0
        for start, end, payload in sorted(self.edits):
            _need(begin <= start, 'overlapping C++ observer sites')
            chunks.extend((self.source[begin:start], payload))
            begin = end
        chunks.append(self.source[begin:])
        return b''.join(chunks)


def _usage_header():
    header = _clock_header().decode('ascii')
    header = header.replace('#include <time.h>', '#include <time.h>\n#include <sys/resource.h>')
    header = header.replace('inline void stamp', '''using Usage = std::array<std::uint64_t, 4>;
static std::array<Usage, 4> usage_stamps{};
static_assert(sizeof(usage_stamps) == 128);

inline void stamp''')
    header = header.replace('    timespec observed{};', '''    if (point >= INPUTS_BEGIN && point <= RECURRENCE_RETURN) {
        rusage observed_usage{};
        if (getrusage(RUSAGE_SELF, &observed_usage) == 0) {
            usage_stamps[point - INPUTS_BEGIN] = {
                1, static_cast<std::uint64_t>(observed_usage.ru_minflt),
                static_cast<std::uint64_t>(observed_usage.ru_majflt),
                static_cast<std::uint64_t>(observed_usage.ru_inblock)
            };
        }
    }
    timespec observed{};''')
    header = header.replace('        std::fclose(out);',
        '        std::fwrite(usage_stamps.data(), sizeof(usage_stamps), 1, out);\n        std::fclose(out);')
    return header.encode('ascii')


def _render(main_ref, file_ref, *, usage=False):
    main = _SourceEdits(main_ref, _raw(main_ref))
    header = _SourceEdits(file_ref, _raw(file_ref))
    entry = b'int main(int argc, char** argv) {\n'
    main.insert('MAIN_ENTER', entry,
        b'    scar::fixed_grid_clock::MainWitness scar_owner_witness;\n'
        b'    scar::fixed_grid_clock::stamp(scar::fixed_grid_clock::MAIN_ENTER);\n', after=True)
    main.mark('RUNTIME_BEGIN', b'        at::set_num_threads(1);\n')
    teams = [line + b'\n' for line in main.source.splitlines() if line.startswith(b'        graph::CpuTeam team(')]
    team, = teams
    main.mark('TEAM_CONSTRUCTOR_RETURN', team, after=True)
    invoke = b'            return invoke(arguments.data(), [&](const auto& inputs) {\n'
    main.rewrite('INVOKE_STATUS_CAPTURE', invoke, invoke.replace(b'return ', b'const auto scar_status = ', 1))
    recurrence = b'                auto result = repeat(inputs, count, team, [](auto, auto&& run) { return run(); });\n'
    main.mark('RECURRENCE_BEGIN', recurrence)
    main.mark('RECURRENCE_RETURN', recurrence, after=True)
    source_returns = [line + b'\n' for line in main.source.splitlines()
                      if line.startswith(b'            }, load_order, ') and line.endswith(b'}, true);')]
    source_return, = source_returns
    main.insert('INVOKE_RETURN', source_return,
        b'            scar::fixed_grid_clock::stamp(scar::fixed_grid_clock::INVOKE_RETURN);\n'
        b'            if (scar_status == 0) scar_owner_witness.destination = arguments.back();\n'
        b'            return scar_status;\n', after=True)
    main.sites['MAIN_END'] = dict(source=main_ref, placement='AUTOMATIC_OWNER_DESTRUCTION',
        byte_offset=None, anchor=main.sites['INVOKE_RETURN']['anchor'])
    header.mark('INPUTS_BEGIN', b'    std::array<std::vector<float>, graph::input_count> backing;\n', source_only=True)
    header.mark('INPUTS_READY', b'    const std::array<at::Tensor, graph::output_count> outputs = runner(inputs);\n',
                source_only=True)
    publish = b'        write_fresh_source_file(destinations[0], outputs[0].const_data_ptr<float>(), sizes[0]);\n'
    header.mark('FILE_PUBLICATION_BEGIN', publish)
    header.mark('FILE_PUBLICATION_RETURN', publish, after=True)
    header.mark('STDOUT_BEGIN', b'    stdout_record(outputs, sizes);\n', source_only=True)
    header.mark('STDOUT_SUBMIT_END', b'    stdout_record(outputs, sizes);\n', after=True, source_only=True)
    return dict(sites={**main.sites, **header.sites}, files={
        _MAIN: main.render(b'#include "runtime/scar_fixed_grid_owner_clock.hpp"\n'),
        _FILE: header.render(b'#include "scar_fixed_grid_owner_clock.hpp"\n'),
        _HEADER: _usage_header() if usage else _clock_header()})


def _project(root, delivery, compiler, *, build, usage=False, mapped_proof=None):
    project_ref = delivery['project']
    project = read_artifact(project_ref)
    description = project['source_entry_cli']
    _need(description.get('source_file_publication') == FRESH_SOURCE_FILE
          and description.get('mapped_input_proof') == mapped_proof
          and project['cxx_standard'] == 20, 'exact source input route and fresh mode 0666 sink required')
    parent = Path(project_ref['uri']).parent
    references = {}
    for name, signature in project['files'].items():
        if name.startswith(('src/', 'include/')) or name in ('CMakeLists.txt', 'launch.sh', LAUNCHER):
            references[name] = dict(uri=str(parent/name), schema='scar.native-project-member', **signature)
    changes = _render(references[_MAIN], references[_FILE], usage=usage)
    _need(_HEADER not in references, 'retained project is already instrumented')
    files = {name: _raw(reference) for name, reference in references.items()}
    files.update(changes['files'])
    binary = project['executable']
    libraries = [Path(project['dependencies'][name]) for name in ('libtorch_cpu.so', 'libc10.so')]
    if build:
        for name, raw in files.items():
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        for name in ('launch.sh', LAUNCHER):
            (root/name).chmod(_mode(parent/name))
        build_native_project(root, Path(binary).name, (compiler['uri'],), expected_libraries=libraries,
                             require_cpu_only=True)
    receipts = {name: _ref(root/name,'scar.native-project-member') for name in files}
    _need(all(_raw(ref) == files[name] for name, ref in receipts.items()), 'C++ sibling changed its pinned project sources')
    diagnostic = _ref(root/binary, 'application/x-executable')
    original = delivery['native_executable']
    _need(diagnostic['sha256'] != original['sha256'] and _raw(diagnostic).startswith(b'\x7fELF')
          and os.access(diagnostic['uri'], os.X_OK) and os.access((root/LAUNCHER), os.X_OK),
          'separate ordinary diagnostic ELF and launcher required')
    loader = _dependencies(subprocess.check_output(['/usr/bin/ldd', diagnostic['uri']], env={}, text=True))
    _need(loader == project['dependencies'] and cpu_executable_dynamic(diagnostic['uri']) ==
          cpu_executable_dynamic(original['uri']), 'instrumented ELF differs from original complete loader closure')
    dependency_refs = {name: None if path is None else dict(path=path, **_content(Path(path)))
                       for name, path in loader.items()}
    return dict(sites=changes['sites'], diagnostic_members=receipts, diagnostic_executable=diagnostic,
                diagnostic_launcher=receipts[LAUNCHER], original_sources={name:references[name] for name in (_MAIN, _FILE)},
                resolved_dependencies=dependency_refs)


def _validate_inputs(rows):
    for row in rows:
        path = Path(row['path'])
        _mode(path)
        _need(_content(path) == {name:row[name] for name in ('sha256', 'size_bytes')},
              'input file changed while observing native phases')


def _capture(folder, command, observation, publication):
    folder.mkdir()
    sidecar = publication.with_name(publication.name + _SUFFIX)
    _need(all(not path.exists() and not path.is_symlink() for path in (publication, sidecar)),
          'exact output or diagnostic sidecar already exists')
    start = clock_gettime_ns(CLOCK_MONOTONIC)
    result = subprocess.run(command, cwd=observation['working_directory'], env=observation['python_provider_environment'],
                            stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
    end = clock_gettime_ns(CLOCK_MONOTONIC)
    (folder/'stdout.bin').write_bytes(result.stdout)
    (folder/'stderr.bin').write_bytes(result.stderr)
    _need(result.returncode == 0 and publication.is_file() and sidecar.is_file(),
          'diagnostic did not publish its exact output and clock sidecar')
    mode = _mode(publication)
    _mode(sidecar)
    shutil.move(publication, folder/'published.f32')
    shutil.move(sidecar, folder/'markers.bin')
    _write(folder/'clock.json', dict(schema=_CLOCK_SCHEMA, launcher_start_ns=start, process_reaped_ns=end,
                                    returncode=result.returncode, publication_mode=mode))


def _sample(folder, command, delivery, ordinal, expected_mode, *, usage=False):
    clock_ref = _ref(folder/'clock.json', _CLOCK_SCHEMA)
    clock = read_artifact(clock_ref)
    stdout, stderr, published, marker_bytes = [_ref(folder/name) for name in
                                            ('stdout.bin', 'stderr.bin', 'published.f32', 'markers.bin')]
    _need({path.name for path in folder.iterdir()} ==
          {'clock.json', 'stdout.bin', 'stderr.bin', 'published.f32', 'markers.bin'}
          and clock['returncode'] == 0 and _raw(stderr) == b''
          and _raw(stdout) == _raw(delivery['validated_source_stdout'])
          and _raw(published) == _raw(delivery['validated_final_tensor'])
          and clock['publication_mode'] == expected_mode == _mode(Path(published['uri'])),
          'observed output, mode or clean native CLI status differs')
    raw = _raw(marker_bytes)
    usage_size = _USAGE.size * len(USAGE_MARKERS) if usage else 0
    _need(len(raw) == _CLOCKS.size + usage_size, 'one complete owner clock record required')
    marks = _CLOCKS.unpack_from(raw)
    bounds = (clock['launcher_start_ns'], *marks, clock['process_reaped_ns'])
    _need(all(type(value) is int and value > 0 for value in bounds) and list(bounds) == sorted(bounds),
          'native phase clocks do not follow source or parent order')
    sample = dict(ordinal=ordinal, argv=command, launcher_start_ns=clock['launcher_start_ns'],
        process_reaped_ns=clock['process_reaped_ns'], returncode=0, publication_mode=expected_mode,
        stdout=stdout, stderr=stderr, publication=published, clock=clock_ref, raw_markers=marker_bytes,
        markers_ns=dict(zip(MARKERS, marks, strict=True)))
    if usage:
        values = tuple(_USAGE.iter_unpack(raw[_CLOCKS.size:]))
        _need(all(valid == 1 for valid, *_ in values)
              and all(list(column) == sorted(column) for column in tuple(zip(*values))[1:]),
              'cumulative process resource counters do not follow observation order')
        sample['process_usage'] = {name: dict(zip(USAGE_FIELDS, fields, strict=True))
                                   for name, (_, *fields) in zip(USAGE_MARKERS, values, strict=True)}
    return sample


def _record(delivery_ref, root, count, compiler, *, build=False):
    delivery = verify_cpu_fixed_grid_cli_delivery(delivery_ref)
    _need(delivery['validation']['exact_publication_mode'] is True and type(count) is int and 1 <= count <= 12,
          'bounded samples of admitted permission-compatible native CLI required')
    _raw(compiler)
    observation = read_artifact(delivery['observation'])
    publication = Path(delivery['argv'][-1])
    expected_mode, = {row['permission_bits'] for row in delivery['publication_modes']}
    _validate_inputs(delivery['file_inputs'])
    if build:
        _need(set(delivery['cpu_ids']) <= os.sched_getaffinity(0), 'exact launch CPU cohort unavailable')
        root.mkdir(parents=True, exist_ok=False)
    project = _project(root/'project', delivery, compiler, build=build)
    original = observation['commands']['NATIVE_CPP']
    command = [project['diagnostic_launcher']['uri'], *delivery['argv']]
    _need(original[1:] == command[1:] and original[0] == delivery['native_launcher']['uri'],
          'diagnostic or original command crosses the exact source CLI argv')
    samples = []
    for index in range(count):
        folder = root/f'sample-{index:02d}'
        if build:
            _capture(folder, command, observation, publication)
        samples.append(_sample(folder, command, delivery, index, expected_mode))
    _validate_inputs(delivery['file_inputs'])
    _need(all(left['process_reaped_ns'] <= right['launcher_start_ns'] for left, right in zip(samples, samples[1:])),
          'independent observation clocks overlap')
    return dict(schema=SCHEMA, schema_version=1, delivery=delivery_ref, project=delivery['project'],
        native_executable=delivery['native_executable'], source=delivery['source'], target=delivery['target'],
        argv=delivery['argv'], steps=delivery['steps'], cpu_ids=delivery['cpu_ids'],
        working_directory=delivery['working_directory'], environment_sha256=delivery['environment_sha256'],
        file_inputs=delivery['file_inputs'], compiler=compiler, cxx_standard=20, **project,
        sample_count=count, samples=samples, clock='LINUX_CLOCK_MONOTONIC_NS',
        windows={name:list(bounds) for name, bounds in WINDOWS.items()}, diagnostic_only=True,
        original_binary_executed=False, diagnostic_binary_executed=True, normalized_output_verified=True,
        exact_exit_streams_file_and_mode=True, clean_benchmark=False, deployment_authorized=False,
        forward_cost_proven=False, physical_duration_bound_proven=False, limits=dict(
            launcher_to_main_includes_exec_dynamic_loader_and_provider_initialization=True,
            team_constructor_return_is_not_all_workers_ready=True, recurrence_is_owner_wall_clock=True,
            file_close_proves_durable_storage=False, stdout_drain_unobserved=True,
            main_end_includes_automatic_local_teardown=True, private_sidecar_written_after_main_end=True,
            end_to_reap_is_not_isolated_process_exit=True))


def measure(delivery_ref, out, *, samples=3, compiler='g++'):
    """Compile and run an exact source-bound diagnostic sibling, with no per-kernel marks."""
    out = Path(out).absolute()
    tool = shutil.which(compiler)
    _need(tool is not None and out.resolve() == out and not out.exists() and not out.is_symlink(),
          'available compiler and fresh canonical result directory required')
    compiler_ref = _ref(Path(tool).resolve(strict=True), 'application/x-executable')
    report = _record(delivery_ref, out, samples, compiler_ref, build=True)
    _write(out/'result.json', report)
    return report


def verify_cpu_fixed_grid_native_lifecycle(reference):
    """Replay source coordinates, diagnostic linkage, effect proofs and raw owner clocks offline."""
    saved = read_artifact(reference)
    path = Path(reference['uri'])
    _need(reference['schema'] == SCHEMA and path.name == 'result.json' and path == path.resolve(strict=True),
          'exact native lifecycle diagnostic report required')
    checked = _record(saved['delivery'], path.parent, saved['sample_count'], saved['compiler'])
    _need(saved == checked, 'retained owner timestamps or C++ diagnostic project differs')
    return saved

"""Measure fixed-grid native lifecycle windows and counters in separate C++20 builds."""

from hashlib import sha256
import os
from pathlib import Path
import shutil
import stat
from statistics import median
from struct import Struct
import subprocess
from time import CLOCK_MONOTONIC, clock_gettime_ns

from scar.codegen.native_provider_bundle import _content, _dependencies, cpu_executable_dynamic
from scar.codegen.aten_recurrence_source import FRESH_SOURCE_FILE
from scar.codegen.native_project import build_native_project
from scar.codegen.cpu_native_route_package import (
    _fixed_grid_route_replay, _output_guards, verify_cpu_fixed_grid_native_route_package,
)
from scar.compiler.aten_entry_delivery import LAUNCHER
from scar.compiler.calibrate_array_entry import _write
from scar.compiler.cpu_fixed_grid_cli_delivery import verify_cpu_fixed_grid_cli_delivery
from scar.compiler.cpu_fixed_grid_mapped_input_study import _input_files
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.cpu_fixed_grid_mapped_input_plan import verify_cpu_fixed_grid_mapped_input_plan


SCHEMA = 'scar.cpu-fixed-grid-native-lifecycle'
SHORT_SCHEMA = 'scar.cpu-fixed-grid-short-native-probe'
COUNTERS_SCHEMA = 'scar.cpu-fixed-grid-input-counters'
_MAPPED_ROUTE = 'PROVIDER_MAP'
_COUNTER_ROUTES = ('OWNED_COPY', _MAPPED_ROUTE)
_COUNTER_CUTS = ('input_ingestion', 'recurrence')
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


def _ordered_samples(samples):
    return all(left['process_reaped_ns'] <= right['launcher_start_ns']
               for left, right in zip(samples, samples[1:]))


def _read_observation(reference, schema, detail, *, require=_need):
    saved = read_artifact(reference)
    path = Path(reference['uri'])
    require(reference['schema'] == schema and path.name == 'result.json' and path.resolve(strict=True) == path,
            detail)
    return saved, path.parent


def _diagnostic_destination(out, compiler, detail):
    out = Path(out).absolute()
    tool = shutil.which(compiler)
    _need(tool is not None and out.resolve() == out and not out.exists() and not out.is_symlink(), detail)
    return out, _ref(Path(tool).resolve(strict=True), 'application/x-executable')


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
    _need(_ordered_samples(samples), 'independent observation clocks overlap')
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
    out, compiler_ref = _diagnostic_destination(out, compiler,
        'available compiler and fresh canonical result directory required')
    report = _record(delivery_ref, out, samples, compiler_ref, build=True)
    _write(out/'result.json', report)
    return report


def verify_cpu_fixed_grid_native_lifecycle(reference):
    """Replay source coordinates, diagnostic linkage, effect proofs and raw owner clocks offline."""
    saved, root = _read_observation(reference, SCHEMA, 'exact native lifecycle diagnostic report required')
    checked = _record(saved['delivery'], root, saved['sample_count'], saved['compiler'])
    _need(saved == checked, 'retained owner timestamps or C++ diagnostic project differs')
    return saved


def _counter_windows(samples):
    result = {}
    for window in _COUNTER_CUTS:
        start, end = WINDOWS[window]
        quantities = {'owner_wall_ns': [row['markers_ns'][end] - row['markers_ns'][start] for row in samples]}
        for field in USAGE_FIELDS:
            quantities[field] = [row['process_usage'][end][field] - row['process_usage'][start][field]
                                 for row in samples]
        result[window] = {name: dict(min=min(values), median=median(values), max=max(values), values=values)
                          for name, values in quantities.items()}
    return result


def _record_counters(plan_ref, root, pairs, compiler, *, capture=False):
    plan = verify_cpu_fixed_grid_mapped_input_plan(plan_ref)
    _need(plan['observation'] is not None and type(pairs) is int and 1 <= pairs <= 12,
          'bounded independent diagnostic for a validated mapped input route required')
    compared = read_artifact(plan['observation'])
    control = read_artifact(compared['control_delivery'])
    input_files, argv = compared['file_inputs'], compared['argv']
    expected_mode = compared['expected_mode']
    _raw(compiler)
    _validate_inputs(input_files)
    if capture:
        _need(set(compared['cpu_ids']) <= os.sched_getaffinity(0), 'exact diagnostic CPU cohort unavailable')
        root.mkdir(parents=True, exist_ok=False)

    owner = dict(working_directory=compared['working_directory'], python_provider_environment=compared['environment'])
    mapped = dict(control, project=compared['mapped_project'], native_executable=compared['mapped_executable'],
                  native_launcher=compared['launchers'][_MAPPED_ROUTE])
    routes = {}
    for name, source, proof in zip(_COUNTER_ROUTES, (control, mapped), (None, plan['proof']), strict=True):
        sibling = _project(root/name, source, compiler, build=capture, usage=True, mapped_proof=proof)
        routes[name] = dict(original_project=source['project'], original_executable=source['native_executable'],
                            sibling=sibling, samples=[])

    publication = Path(argv[-1])
    ordered = []
    for pair in range(pairs):
        for name in _COUNTER_ROUTES if pair % 2 == 0 else reversed(_COUNTER_ROUTES):
            row = routes[name]
            folder = root/f'{pair:02}-{name}'
            command = [row['sibling']['diagnostic_launcher']['uri'], *argv]
            if capture:
                _capture(folder, command, owner, publication)
            sample = _sample(folder, command, control, pair, expected_mode, usage=True)
            row['samples'].append(sample)
            ordered.append(sample)
    _need(_ordered_samples(ordered), 'route diagnostic process windows overlap')
    _validate_inputs(input_files)
    for row in routes.values():
        row['windows'] = _counter_windows(row['samples'])

    return dict(schema=COUNTERS_SCHEMA, schema_version=1, mapped_plan=plan_ref, observation=plan['observation'],
        control_delivery=compared['control_delivery'], source=compared['source'], target=compared['target'],
        argv=argv, cpu_ids=compared['cpu_ids'], working_directory=compared['working_directory'],
        environment_sha256=compared['environment_sha256'], compiler=compiler, sample_pairs=pairs, routes=routes,
        scope='PROCESS_WIDE_RUSAGE_SELF_AT_OWNER_SOURCE_PHASE_BOUNDARIES',
        normal_completion_stdout_stderr_publication_and_mode_verified=True,
        clean_benchmark=False, physical_first_touch_attributed=False, physical_io_bytes_measured=False,
        page_cache_condition_controlled=False, forward_duration_bound_proven=False, deployment_authorized=False)


def measure_cpu_fixed_grid_input_counters(plan_ref, out, *, pairs=3, compiler='g++'):
    """Capture process counters for the two admitted input routes."""
    out, compiler_ref = _diagnostic_destination(out, compiler,
        'available compiler and fresh canonical output directory required')
    report = _record_counters(plan_ref, out, pairs, compiler_ref, capture=True)
    _write(out/'result.json', report)
    return report


def verify_cpu_fixed_grid_input_counters(reference):
    saved, root = _read_observation(reference, COUNTERS_SCHEMA, 'exact route diagnostic report required')
    expected = _record_counters(saved['mapped_plan'], root, saved['sample_pairs'], saved['compiler'])
    _need(expected == saved, 'retained paired source counter observations differ')
    return saved


def _need_short(condition, reason):
    if not condition:
        raise ValueError('fixed-grid short native probe: ' + reason)


def _short_binding(package_ref, oracle_ref):
    from scar.compiler.observe_cpu_fixed_grid_python_lifecycle import (
        SHORT_SCHEMA as SHORT_PYTHON_SCHEMA, STEP_COUNTS, verify_short_python,
    )

    _need_short(oracle_ref['schema'] == SHORT_PYTHON_SCHEMA, 'exact independent short-count Python oracle required')
    with _fixed_grid_route_replay():
        package = verify_cpu_fixed_grid_native_route_package(package_ref)
        oracle = verify_short_python(oracle_ref)
    _need_short(package['selected_runtime'] == _MAPPED_ROUTE and package['inputs_must_remain_stable'] is True
          and package['deployment_authorized'] is False and oracle['package'] == package_ref
          and oracle['source'] == package['source'] and oracle['target'] == package['hardware']
          and oracle['cpu_ids'] == package['cpu_ids'] and oracle['working_directory'] == package['working_directory']
          and oracle['environment_sha256'] == package['environment_sha256']
          and oracle['step_counts'] == list(STEP_COUNTS) and set(oracle['cases']) == {str(count) for count in STEP_COUNTS},
          'mapped native route and Python oracle scopes differ')
    plan = read_artifact(package['plan'])
    proof_ref = plan['proof']
    abi = read_artifact(read_artifact(proof_ref)['abi'])
    argv = abi['argv']
    step, output = argv['steps'] - 1, argv['output'] - 1
    _need_short(argv['count'] == len(package['argv']) and output == package['publication']['output_index']
          and step != output and abi['domain']['normal_completion_only'] is True
          and abi['domain']['step_count_min'] <= min(STEP_COUNTS) <= max(STEP_COUNTS) <= abi['domain']['step_count_max'],
          'short counts or publication slot are outside the exact source ABI')
    original = read_artifact(read_artifact(package['owned_copy_delivery'])['observation'])['commands']['ORIGINAL_PYTHON']
    prefix = original[:-argv['count']]
    for count in STEP_COUNTS:
        row = oracle['cases'][str(count)]
        value = row['oracle']
        command = value['command']
        args = command[len(prefix):]
        expected = list(package['argv'])
        expected[step] = str(count)
        expected[output] = args[output]
        _need_short(command[:len(prefix)] == prefix and args == expected == row['argv'] and Path(args[output]).is_absolute()
              and args[output] != package['publication']['path']
              and value['mode'] == package['publication']['expected_mode']
              and value['file']['size_bytes'] == package['publication']['expected_file']['size_bytes']
              and not _raw(value['stderr']), 'short-count original argv, mode or final file oracle differs')
    return package, oracle, proof_ref, step, output


def _short_sibling(package, proof_ref, root, compiler, counters_ref, *, build):
    route = package['routes'][_MAPPED_ROUTE]
    material = dict(project=route['project'], native_executable=route['executable'])
    if counters_ref is None:
        sibling = _project(root/'native-probe', material, compiler, build=build, mapped_proof=proof_ref)
        return sibling, False
    _need_short(counters_ref['schema'] == COUNTERS_SCHEMA and Path(counters_ref['uri']).name == 'result.json',
          'exact retained counter sibling authority required')
    counters = read_artifact(counters_ref)
    retained = counters['routes'][_MAPPED_ROUTE]
    _need_short(counters['mapped_plan'] == package['plan'] and counters['observation'] == package['observation']
          and counters['control_delivery'] == package['owned_copy_delivery']
          and counters['source'] == package['source'] and counters['target'] == package['hardware']
          and counters['cpu_ids'] == package['cpu_ids'] and counters['working_directory'] == package['working_directory']
          and counters['argv'] == package['argv'] and counters['environment_sha256'] == package['environment_sha256']
          and counters['compiler'] == compiler and counters['clean_benchmark'] is False
          and retained['original_project'] == route['project'] and retained['original_executable'] == route['executable'],
          'retained marked C++ executable does not cover the same mapped route and context')
    sibling = _project(Path(counters_ref['uri']).parent/_MAPPED_ROUTE, material, compiler, build=False,
                       usage=True, mapped_proof=proof_ref)
    _need_short(sibling == retained['sibling'], 'retained diagnostic C++ source, binary or loader changed')
    return sibling, True


def _short_report(package_ref, oracle_ref, root, count, compiler, counters_ref, *, capture=False):
    package, oracle, proof_ref, step, output = _short_binding(package_ref, oracle_ref)
    _need_short(type(count) is int and 1 <= count <= 12 and root.resolve() == root
          and not root.is_relative_to(Path(package_ref['uri']).parent)
          and not root.is_relative_to(Path(oracle_ref['uri']).parent)
          and all(not root.is_relative_to(route['project_root']) for route in package['routes'].values())
          and all(not root.is_relative_to(Path(row['path']).parent) for row in package['file_inputs']),
          'bounded samples and independent output directory required')
    _raw(compiler)
    if capture:
        _need_short(set(package['cpu_ids']) <= os.sched_getaffinity(0), 'the exact launch CPU cohort is unavailable')
        root.mkdir(parents=True, mode=0o700, exist_ok=False)
    sibling, usage = _short_sibling(package, proof_ref, root, compiler, counters_ref, build=capture)
    owner = dict(working_directory=package['working_directory'], python_provider_environment=package['source_environment'])
    expected_mode = package['publication']['expected_mode']
    input_snapshots = _input_files(package)
    cases, ordered = {}, []
    outputs = root/'private-output-slots'
    if capture:
        outputs.mkdir(mode=0o700)
    for steps in oracle['step_counts']:
        expected = oracle['cases'][str(steps)]['oracle']
        check = dict(validated_source_stdout=expected['stdout'], validated_final_tensor=expected['file'])
        rows = []
        for index in range(count):
            name = f'{steps:02d}-sample-{index:02d}'
            publish = outputs/f'{name}.f32'
            args = list(package['argv'])
            args[step], args[output] = str(steps), str(publish)
            command = [sibling['diagnostic_launcher']['uri'], *args]
            folder = root/name
            if capture:
                _need_short(not _output_guards(publish, expected_mode), 'private output is occupied or its mode premise changed')
                _capture(folder, command, owner, publish)
            row = _sample(folder, command, check, index, expected_mode, usage=usage)
            markers = row['markers_ns']
            first, last = markers['RECURRENCE_BEGIN'], markers['RECURRENCE_RETURN']
            row['windows_ns'] = dict(before_recurrence=first - row['launcher_start_ns'],
                recurrence=last - first, after_recurrence=row['process_reaped_ns'] - last)
            _need_short(all(value > 0 for value in row['windows_ns'].values())
                  and _raw(row['stderr']) == _raw(expected['stderr']), 'phase clock or original stderr differs')
            rows.append(row)
            ordered.append(row)
        cases[str(steps)] = dict(steps=steps, oracle=expected, samples=rows)
    _need_short(_input_files(package) == input_snapshots and not any(outputs.iterdir())
          and _ordered_samples(ordered),
          'input identity, privately staged output or sample order changed')
    return dict(schema=SHORT_SCHEMA, schema_version=1, status='EXACT_SHORT_NATIVE_LIFECYCLE_OBSERVED',
        package=package_ref, python_probe=oracle_ref, project=package['selected_project'],
        native_executable=package['routes'][_MAPPED_ROUTE]['executable'], mapped_proof=proof_ref,
        source=package['source'], target=package['hardware'], cpu_ids=package['cpu_ids'],
        working_directory=package['working_directory'], environment_sha256=package['environment_sha256'],
        argv=package['argv'], file_inputs=package['file_inputs'], input_snapshots=input_snapshots,
        step_counts=list(oracle['step_counts']), cases=cases, sample_count=count, compiler=compiler,
        counters_ref=counters_ref, sibling=sibling, clock='LINUX_CLOCK_MONOTONIC_NS',
        exact_exit_streams_file_and_mode_verified=True, diagnostic_only=True, clean_benchmark=False,
        instrumented_rusage_at_four_cuts=usage, physical_duration_bound_proven=False,
        deployment_authorized=False, target_executed=True,
        limitations=['BEFORE_RECURRENCE_INCLUDES_LOADER_AND_INPUT_INGRESS',
                     'AFTER_RECURRENCE_INCLUDES_PUBLICATION_AND_PROBE_RECORDER_BEFORE_REAP',
                     'PER_KERNEL_AND_HARDWARE_EVENTS_NOT_MEASURED'])


def observe_short_native(package_ref, oracle_ref, out, *, samples=3, counters_ref=None):
    """Measure three independent native lifecycle windows per declared short count."""
    out = Path(out).absolute()
    _need_short(out.resolve() == out and not out.exists() and not out.is_symlink(), 'fresh canonical output required')
    if counters_ref is not None:
        compiler = read_artifact(counters_ref)['compiler']
    else:
        tool = shutil.which('g++')
        _need_short(tool is not None, 'C++20 compiler required')
        compiler = _ref(Path(tool).resolve(strict=True), 'application/x-executable')
    report = _short_report(package_ref, oracle_ref, out, samples, compiler, counters_ref, capture=True)
    _write(out/'result.json', report)
    return report


def verify_short_native(reference):
    """Replay exact original outputs, source modifications and native clock records cold."""
    saved, root = _read_observation(reference, SHORT_SCHEMA, 'exact short native lifecycle receipt required',
                                    require=_need_short)
    expected = _short_report(saved['package'], saved['python_probe'], root, saved['sample_count'],
                       saved['compiler'], saved['counters_ref'])
    _need_short(saved == expected, 'saved native observations differ from cold replay')
    return saved

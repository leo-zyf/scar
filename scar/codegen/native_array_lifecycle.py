"""Observe an instrumented sibling of an independently retained C++20 array entry."""

import json
import os
from pathlib import Path
from struct import unpack
import subprocess
import sys

from scar.codegen.native_cpu_elf import _dependencies
from scar.codegen.native_cpu_elf import cpu_executable_dynamic
from scar.codegen.native_project import build_native_project
from scar.compiler.cpu_entry_replay import _build_capacity_guard, _capacity_guard_receipt, _published
from scar.compiler.cpu_entry_replay_adapters import _match_json_report, array_tasks
from scar.compiler.entry_delivery import _environment_digest
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.deployment_cost import _pointer
from scar.runtime.cpu_paired_runner import capacity_receipt


SCHEMA = 'scar.cpu-native-array-lifecycle-observation'
SITE_SCHEMA = 'scar.cpu-native-phase-sites'
MARKERS = ('MAIN_ENTER', 'DISPATCH_BEGIN', 'DISPATCH_RETURN', 'PUBLICATION_BEGIN', 'PUBLICATION_END')
WINDOWS = dict(startup=('started_ns', 'MAIN_ENTER'), guards=('MAIN_ENTER', 'DISPATCH_BEGIN'),
    dispatch_join=('DISPATCH_BEGIN', 'DISPATCH_RETURN'), post_dispatch=('DISPATCH_RETURN', 'PUBLICATION_BEGIN'),
    publication=('PUBLICATION_BEGIN', 'PUBLICATION_END'), teardown=('PUBLICATION_END', 'reaped_ns'))
_SOURCE = 'src/main.cpp'
_HEADER = 'include/scar_generated/cpu_array_lifecycle.hpp'
_RAW = 'application/octet-stream'
_PACKAGE = Path(__file__).resolve().parents[1]


def _need(ok, why):
    if not ok:
        raise ValueError('native array lifecycle: ' + why)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _raw(reference):
    return read_artifact(reference, json_document=False)


def _inputs(reference):
    from scar.analysis.cpu_native_phase_sites import verify_cpu_native_phase_sites

    sites = verify_cpu_native_phase_sites(reference)
    graph = read_artifact(sites['lifecycle_graph'])
    capacity_ref = graph['capacity_contract']
    capacity = read_artifact(capacity_ref)
    _need(capacity['status'] == 'CANDIDATE_GUARD_ADMISSIBLE' and capacity['option_id'] == sites['option_id']
          and capacity['evidence']['runtime'] == sites['runtime'] and capacity['evidence']['schedule'] == sites['schedule'],
          'phase sites lack the exact admitted native entry and address-space cap')
    ledger = read_artifact(capacity['ledger'])
    option, = (row for row in ledger['options'] if row['option_id'] == sites['option_id'])
    proof, = option['legality_evidence']
    choice = _pointer(read_artifact(proof['artifact']), proof['locator'])['generated_option']['proof']['emission_plan']
    compiled = choice['contract']['profile']
    spec = array_tasks(compiled, compiled, read_artifact(compiled['provider']))
    candidate = spec['arms']['candidate']
    launch = capacity['native_launch']
    leaf, = spec['files']
    _need(compiled['runtime'] == sites['runtime'] and candidate['executable'] == sites['binary'] == launch['executable']
          and candidate['argv'] == [*launch['argv'][:-1], '{output_dir}/' + leaf]
          and candidate['output_slot'] == launch['output_index'] + 1 == len(candidate['argv']) - 1
          and spec['cpu_ids'] == launch['cpu_ids'] and spec['cwd'] == launch['cwd']
          and _environment_digest(candidate['environment']) == launch['environment_sha256'],
          'retained native invocation, context or publication slot differs')
    runtime = read_artifact(sites['runtime'])
    schedule = read_artifact(sites['schedule'])
    compiler = read_artifact(schedule['thread_evidence'])['compiler_binary']
    _raw(compiler)
    return sites, capacity_ref, capacity, spec, runtime, compiler


def render_instrumented_source(source, positions):
    """Insert only standalone marker statements at original byte offsets."""
    _need(tuple(row['id'] for row in positions) == MARKERS, 'incomplete lifecycle marker sequence')
    result = [b'#include "cpu_array_lifecycle.hpp"\n']
    previous = 0
    for row in positions:
        offset = row['byte_offset']
        indentation = row['indentation'].encode('ascii')
        _need(type(offset) is int and previous < offset < len(source) and source[offset-1:offset] == b'\n'
              and indentation and not indentation.strip(b' ') and source[offset:].startswith(indentation),
              'marker does not start its exact generated statement line')
        name = row['id']
        statement = ('scar::entry_lifecycle::Witness scar_lifecycle;' if name == 'MAIN_ENTER' else
                     f'scar_lifecycle.mark(scar::entry_lifecycle::Marker::{name});')
        result.extend((source[previous:offset], indentation, statement.encode('ascii'), b'\n'))
        previous = offset
    result.append(source[previous:])
    return b''.join(result)


def _members(runtime, sites):
    _need(runtime['files'][_SOURCE] == sites['cpp_source'] and runtime['files'][runtime['binary']] == sites['binary']
          and _HEADER not in runtime['files'], 'generated source or binary differs from the phase site package')
    originals = {}
    for name, reference in runtime['files'].items():
        path = Path(name)
        _need(not path.is_absolute() and '..' not in path.parts and path.as_posix() == name
              and str(Path(sites['runtime']['uri']).parent / path) == reference['uri'], 'runtime member path differs')
        if name != runtime['binary']:
            originals[name] = _raw(reference)
    originals[_SOURCE] = render_instrumented_source(originals[_SOURCE], sites['sites'])
    originals[_HEADER] = (_PACKAGE/'native/cpu_array_lifecycle.hpp').read_bytes()
    return originals


def _project(root, runtime, sites, compiler, *, build=False):
    package = root/'project'
    members = _members(runtime, sites)
    if build:
        for name, raw in members.items():
            destination = package / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        libraries = [package/name for name in members if name.startswith('provider/objects/')]
        build_native_project(package, Path(runtime['binary']).name, (compiler['uri'],), expected_libraries=libraries)
    references = {name: _ref(package/name, 'scar.native-project-member') for name in members}
    _need(all(reference['uri'] == str(package/name) and _raw(reference) == members[name]
              for name, reference in references.items()), 'instrumented source, header, unchanged build input or library differs')
    binary = _ref(package/runtime['binary'], 'scar.native-project-member')
    _need(binary['uri'] == str(package/runtime['binary']) and _raw(binary).startswith(b'\x7fELF')
          and binary['sha256'] != sites['binary']['sha256']
          and cpu_executable_dynamic(binary['uri']) == runtime['linkage']['dynamic'],
          'sibling executable or native loader declarations differ')
    return binary, references


def _loader(executable, spec):
    listing = subprocess.check_output(['/usr/bin/ldd', executable['uri']], cwd=spec['cwd'],
        env=spec['arms']['candidate']['environment'], text=True)
    return {key: None if path is None else _ref(path, 'scar.elf')
            for key, path in _dependencies(listing).items()}


def _same_loader(original, sibling, runtime):
    _need(type(original) is type(sibling) is dict and original.keys() == sibling.keys()
          and set(runtime['linkage']['dynamic']['needed']) <= set(original), 'loader dependency closure differs')
    for key, left in original.items():
        right = sibling[key]
        _need((left is None and right is None) or (left is not None and right is not None and _raw(left) == _raw(right)),
              'original and diagnostic executable resolve different loader images')


def _execution(sibling, spec, guard, limit, folder):
    candidate = spec['arms']['candidate']
    leaf, = spec['files']
    argv = ['/usr/bin/taskset', '-c', ','.join(map(str, spec['cpu_ids'])), '/usr/bin/timeout', '-k', '2', '120',
            sibling['uri'], *candidate['argv'][1:-1], str(folder/'published'/leaf)]
    cap = capacity_receipt(argv, (), limit, 199, native_capacity_guard_executable=guard['executable']['uri'])
    return dict(argv=cap['launch_argv'], cwd=spec['cwd'], environment=candidate['environment']), cap


def _samples(root, spec, binary, guard, limit, raw_rows, warmups, count):
    _need(type(raw_rows) is list and len(raw_rows) == warmups + count, 'observed sample count differs')
    recipe = spec['json_report']
    expected = _raw(recipe['oracle'])
    leaf, = spec['files']
    rows = []
    last = 0
    for index, raw in enumerate(raw_rows):
        folder = root/f'sample-{index:02d}'
        launch, cap = _execution(binary, spec, guard, limit, folder)
        stdout, stderr, markers, receipt = (_ref(folder/name, _RAW) for name in
                                             ('stdout.bin', 'stderr.bin', 'markers.bin', 'capacity.bin'))
        publications = _published(folder)
        _need(publications.keys() == {leaf}, 'diagnostic sibling changed the declared publication set')
        published = publications[leaf]
        _need(type(raw) is dict and raw.keys() == {'status', 'started_ns', 'reaped_ns'}
              and type(raw['status']) is int and raw['status'] == 0
              and not _raw(stderr) and _match_json_report(_raw(stdout), _raw(published), recipe, expected),
              'diagnostic sibling differs from retained stdout, publication or exit contract')
        _need(_raw(receipt) == f'{limit} {limit}\n'.encode('ascii'), 'pre-exec address-space receipt differs')
        stamps = _raw(markers)
        start, end = raw['started_ns'], raw['reaped_ns']
        _need(len(stamps) == len(MARKERS)*8 and type(start) is type(end) is int and last <= start < end,
              'invalid private marker bytes or sampler clock window')
        values = unpack('=5Q', stamps)
        bounds = (start, *values, end)
        _need(start > 0 and all(left <= right for left, right in zip(bounds, bounds[1:])),
              'phase timestamps do not follow the anchored owner order')
        last = end
        rows.append(dict(ordinal=index, warmup=index < warmups, started_ns=start, reaped_ns=end,
            markers_ns=dict(zip(MARKERS, values, strict=True)), stdout=stdout, stderr=stderr, published=published,
            marker_bytes=markers, capacity_readback=receipt, capacity_guard=cap, argv=launch['argv']))
    return rows


def _context(warmups, samples):
    _need(type(warmups) is type(samples) is int and 0 <= warmups <= 4 and 1 <= samples <= 12,
          'bounded nonnegative warmups and at least one diagnostic sample required')


def _report(root, reference, inputs, binary, members, guard, loaders, interpreter, warmups, samples):
    sites, cap_ref, capacity, spec, runtime, compiler = inputs
    _same_loader(loaders['original'], loaders['diagnostic'], runtime)
    sampler = _ref(root/'sampler.py', 'scar.python-source')
    clocks = _ref(root/'clock-samples.json', 'scar.cpu-native-array-lifecycle-clocks')
    _raw(interpreter)
    rows = _samples(root, spec, binary, guard, capacity['candidate_address_space_limit_bytes'],
                    read_artifact(clocks)['rows'], warmups, samples)
    return dict(schema=SCHEMA, schema_version=1, scope='INSTRUMENTED_NATIVE_ENTRY_SIBLING', phase_sites=reference,
        graph=sites['lifecycle_graph'], option_id=sites['option_id'], scope_sha256=sites['scope_sha256'],
        runtime=sites['runtime'], original_executable=sites['binary'], diagnostic_executable=binary,
        cpu_ids=spec['cpu_ids'], cwd=spec['cwd'], environment_sha256=capacity['native_launch']['environment_sha256'],
        clock='LINUX_CLOCK_MONOTONIC_NS', clock_samples=clocks, samples=rows, warmup_count=warmups, sample_count=samples,
        windows={name: list(bounds) for name, bounds in WINDOWS.items()}, output_oracle=spec['json_report']['oracle'],
        build=dict(compiler=compiler, cxx_standard=20, project_members=members, loader_images=loaders),
        sampler=dict(source=sampler, interpreter=interpreter), capacity_contract=cap_ref, capacity_guard=guard,
        instrumented=True, original_workload_executed=False, original_binary_executed=False,
        diagnostic_binary_executed=True, normalized_output_verified=True, clean_benchmark=False,
        whole_entry_cost_bound_proven=False, deployment_authorized=False,
        phase_limits=dict(pre_exec_guard_grouped_in_startup=True, dispatch_includes_join=True,
            buffered_stdout_flush_may_appear_in_teardown=True, marker_publication_in_teardown=True))


def _write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def measure(phase_sites_ref, out, *, warmups=0, samples=3):
    """Build one sibling and retain independent normal-path phase observations."""
    _context(warmups, samples)
    reference = phase_sites_ref if type(phase_sites_ref) is dict else _ref(phase_sites_ref, SITE_SCHEMA)
    inputs = _inputs(reference)
    sites, _, capacity, spec, runtime, compiler = inputs
    _need(sys.platform == 'linux' and set(spec['cpu_ids']) <= os.sched_getaffinity(0), 'admitted Linux CPU cohort is unavailable')
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    binary, members = _project(root, runtime, sites, compiler, build=True)
    loaders = dict(original=_loader(sites['binary'], spec), diagnostic=_loader(binary, spec))
    _same_loader(loaders['original'], loaders['diagnostic'], runtime)
    guard = _build_capacity_guard(root)
    sampler = root/'sampler.py'
    sampler.write_bytes((_PACKAGE/'runtime/native_array_lifecycle_sampler.py').read_bytes())
    interpreter = _ref(sys.executable, 'application/x-executable')
    limit = capacity['candidate_address_space_limit_bytes']
    directories = [root/f'sample-{index:02d}' for index in range(warmups + samples)]
    request = [(_execution(binary, spec, guard, limit, folder)[0], str(folder)) for folder in directories]
    sampled = subprocess.check_output([interpreter['uri'], '-I', '-S', '-B', str(sampler)], cwd=root, env={},
                                      input=json.dumps(request), text=True)
    _write(root/'clock-samples.json', dict(schema='scar.cpu-native-array-lifecycle-clocks', rows=json.loads(sampled)))
    report = _report(root, reference, inputs, binary, members, guard, loaders, interpreter, warmups, samples)
    _write(root/'result.json', report)
    return report


def check(path):
    """Cold-replay material, scope, marker and output proof without running either ELF."""
    path = Path(path).resolve(strict=True)
    saved = read_artifact(_ref(path, SCHEMA))
    reference = saved['phase_sites']
    inputs = _inputs(reference)
    sites, _, _, _, runtime, compiler = inputs
    warmups, samples = saved['warmup_count'], saved['sample_count']
    _context(warmups, samples)
    binary, members = _project(path.parent, runtime, sites, compiler)
    cap = saved['capacity_guard']
    guard = _capacity_guard_receipt(path.parent, cap['compiler'], cap['source'], cap['executable'])
    expected = _report(path.parent, reference, inputs, binary, members, guard, saved['build']['loader_images'],
                       saved['sampler']['interpreter'], warmups, samples)
    _need(json.dumps(saved, sort_keys=True) == json.dumps(expected, sort_keys=True), 'stored lifecycle observation differs')
    return saved

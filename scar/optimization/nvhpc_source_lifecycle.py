"""Collect source-bound lifecycle component samples, independently of clean A/B."""

import argparse
import ast
import json
from pathlib import Path
from statistics import median
import subprocess
from tempfile import TemporaryDirectory
from time import monotonic_ns

from scar.analysis.fixed_grid_entry_abi import _span
from scar.codegen.native_provider_bundle import _content, _dependencies
from scar.codegen.native_project import build_native_project
from scar.compiler.compile_exported_aten import _calibration_paths, _write_record
from scar.compiler.compile_exported_nvhpc import _compare, check_nvhpc_project
from scar.compiler.nvhpc_source_validation import _context, _receipt
from scar.compiler.operator_stack import load_operator_stack, nvidia_stack_build, verify_operator_stack_linkage
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.nvhpc-source-lifecycle-observation'
_ARMS = ('ORIGINAL', 'NATIVE')
_PREFIX = 'SCAR_LIFECYCLE '
_LIMITATIONS = {
    'startup': 'Parent launch, interpreter or ELF loader and probe bootstrap are grouped.',
    'imports': 'Python imports may initialize providers and workers. Their internal costs are grouped.',
    'loading': 'Logical load operations include their allocation or mapping. Deferred mmap faults can occur in execution.',
    'execution': 'Dispatch, provider internals and page faults are included. Internal provider workspace and contention are unmeasured.',
    'publication': 'Logical conversion, write and close are included. Writeback and durability are unmeasured.',
    'teardown': 'Source releases, process exit, probe serialization and parent wait are grouped.',
    'unattributed': 'Gaps between timed statements and clock bookkeeping remain unattributed.',
}


def _python_probe(path, contract):
    raw = path.read_bytes()
    module = ast.parse(raw, filename=str(path))
    stem = '_scar_lifecycle_' + contract['source']['sha256'][:12]
    if stem.encode() in raw:
        raise ValueError('lifecycle probe names overlap the source entry')

    def mark(phase, slot, edge):
        return ast.Expr(ast.Call(ast.Name(stem + '_mark', ast.Load()),
            [ast.Constant(phase), ast.Constant(slot), ast.Constant(edge)], []))

    phases = {'PRIVATE_FILE_MAPPING_AND_TENSOR_VIEW': 'loading', 'FUNCTIONAL_MM': 'execution',
              'SYNC_FILE_CONTENT_PUBLICATION': 'publication'}
    slots = dict.fromkeys(phases.values(), 0)
    by_span = {}
    for node in contract['graph']['nodes']:
        if node['kind'] in phases:
            phase = phases[node['kind']]
            by_span[tuple(node['source_span'])] = phase, slots[phase]
            slots[phase] += 1
    function, = (row for row in module.body if type(row) is ast.FunctionDef
                 and row.name == contract['source']['function'])
    instrumented, seen = [], set()
    for statement in function.body:
        span = _span(statement.value) if type(statement) in (ast.Assign, ast.Expr) else None
        if span in by_span:
            phase, slot = by_span[span]
            seen.add(span)
            if phase == 'loading' and slot == 0:
                instrumented.append(mark('guards', 0, 'end'))
            instrumented.extend((mark(phase, slot, 'begin'), statement, mark(phase, slot, 'end')))
        else:
            instrumented.append(statement)
    if seen != by_span.keys():
        raise ValueError('lifecycle spans differ from the original source activity graph')
    doc = ast.get_docstring(function, clean=False) is not None
    instrumented.insert(int(doc), mark('guards', 0, 'begin'))
    function.body = instrumented
    imports = [index for index, row in enumerate(module.body) if type(row) in (ast.Import, ast.ImportFrom)]
    module.body.insert(imports[-1] + 1, mark('imports', 0, 'end'))
    module.body.insert(imports[0], mark('imports', 0, 'begin'))
    # The source grammar allows only the pinned imports, definitions and entry call at module scope.
    source = ast.unparse(ast.fix_missing_locations(module))
    prelude = f'''import sys as {stem}_sys
from time import monotonic_ns as {stem}_clock
{stem}_events = []
def {stem}_mark(phase, slot, edge):
    {stem}_events.append((phase, slot, edge, {stem}_clock()))
{stem}_mark('process', 0, 'entered')
{stem}_sys.argv[0] = {str(path)!r}
{stem}_sys.path[0] = {str(path.parent)!r}
__file__ = {str(path)!r}
'''
    trailer = f'''{stem}_mark('process', 0, 'leaving')
for {stem}_row in {stem}_events:
    {stem}_sys.stderr.write({_PREFIX!r} + ' '.join(map(str, {stem}_row)) + '\\n')
'''
    return prelude + source + '\n' + trailer


_NATIVE_MARKER = r'''
#include <time.h>

namespace {
struct LifecycleEvent { const char* phase; std::size_t index; const char* edge; long long ns; };
constexpr auto event_capacity = 8 + 2 * matrix_plan::input_bytes.size()
    + 6 * matrix_plan::products.size() + 2 * matrix_plan::outputs.size();
std::array<LifecycleEvent, event_capacity> lifecycle{};
std::size_t recorded = 0;

void mark_lifecycle(const char* phase, std::size_t index, const char* edge) {
    timespec time{};
    if (recorded >= lifecycle.size() || clock_gettime(CLOCK_MONOTONIC, &time) != 0) {
        throw std::runtime_error("cannot record the lifecycle monotonic clock");
    }
    lifecycle[recorded++] = {phase, index, edge, 1000000000LL * time.tv_sec + time.tv_nsec};
}

void write_lifecycle() {
    for (std::size_t index = 0; index < recorded; ++index) {
        const auto& event = lifecycle[index];
        std::fprintf(stderr, "SCAR_LIFECYCLE %s %zu %s %lld\n", event.phase, event.index, event.edge, event.ns);
    }
}
}
'''


def _native_probe(source):
    anchors = {
        'static_assert(sizeof(blasint)': _NATIVE_MARKER + '\n{anchor}',
        'int main(int argc, char** argv) try {': '{anchor}\n'
            '    mark_lifecycle("process", 0, "entered");\n    mark_lifecycle("guards", 0, "begin");',
        '    openblas_set_num_threads(matrix_plan::threads);':
            '    mark_lifecycle("guards", 0, "end");\n    mark_lifecycle("worker_lifecycle", 0, "begin");\n{anchor}',
        '    std::array<Buffer, matrix_plan::buffers> buffers;':
            '    mark_lifecycle("worker_lifecycle", 0, "end");\n{anchor}',
        '        buffers[index] = load(argv[1 + index], matrix_plan::input_bytes[index]);':
            '        mark_lifecycle("loading", index, "begin");\n{anchor}\n        mark_lifecycle("loading", index, "end");',
        '        auto result = std::make_unique_for_overwrite<float[]>(std::size_t(op.m) * op.n);':
            '        mark_lifecycle("allocation", index, "begin");\n{anchor}\n'
            '        mark_lifecycle("allocation", index, "end");\n        mark_lifecycle("execution", index, "begin");',
        '        buffers[op.result] = std::move(result);':
            '        mark_lifecycle("execution", index, "end");\n        mark_lifecycle("release", index, "begin");\n{anchor}',
        '            buffers[matrix_plan::releases[rear]].reset();\n        }':
            '{anchor}\n        mark_lifecycle("release", index, "end");',
        '        publish(destinations[index], buffers[matrix_plan::outputs[index]], matrix_plan::output_bytes[index]);':
            '        mark_lifecycle("publication", index, "begin");\n{anchor}\n'
            '        mark_lifecycle("publication", index, "end");',
        '    return 0;\n} catch (const std::exception& error)':
            '    mark_lifecycle("process", 0, "leaving");\n    write_lifecycle();\n{anchor}',
    }
    for anchor, decoration in anchors.items():
        if source.count(anchor) != 1:
            raise ValueError('native source no longer has its exact lifecycle observation boundary: ' + anchor)
        source = source.replace(anchor, decoration.replace('{anchor}', anchor))
    return source


def _markers(contract, recipe, mode):
    rows = [('process', 0, 'entered')]

    def add(phase, index=0):
        rows.extend((phase, index, edge) for edge in ('begin', 'end'))

    if mode == 'ORIGINAL':
        add('imports')
        add('guards')
    else:
        add('guards')
        add('worker_lifecycle')
    for index in range(len(contract['inputs'])):
        add('loading', index)
    for index in range(len(recipe['operations'])):
        if mode == 'NATIVE':
            add('allocation', index)
        add('execution', index)
        if mode == 'NATIVE':
            add('release', index)
    for index in range(len(contract['outputs'])):
        add('publication', index)
    rows.append(('process', 0, 'leaving'))
    return rows


def _parse_samples(raw):
    events = []
    for line in raw.decode('ascii').splitlines():
        if not line.startswith(_PREFIX):
            raise ValueError('lifecycle probe changed original stderr')
        phase, index, edge, timestamp = line.removeprefix(_PREFIX).split()
        events.append(dict(phase=phase, index=int(index), edge=edge, monotonic_ns=int(timestamp)))
    return events


def _phase_times(start, end, events, markers):
    if [(row['phase'], row['index'], row['edge']) for row in events] != markers:
        raise ValueError('lifecycle probe event order differs from the source activity graph')
    times = [row['monotonic_ns'] for row in events]
    if any(type(ns) is not int for ns in (start, *times, end)) or not start <= times[0] <= times[-1] <= end:
        raise ValueError('lifecycle probe times are outside its monotonic process interval')
    if times != sorted(times):
        raise ValueError('lifecycle probe clock order differs from its activity order')
    phases = dict(startup=times[0] - start, teardown=end - times[-2])
    for left, right in zip(events[1:-1:2], events[2:-1:2], strict=True):
        phase = 'teardown' if left['phase'] == 'release' else left['phase']
        phases[phase] = phases.get(phase, 0) + right['monotonic_ns'] - left['monotonic_ns']
    phases['unattributed'] = end - start - sum(phases.values())
    if phases['unattributed'] < 0:
        raise ValueError('lifecycle probe counted a phase twice')
    return phases


def _summary(runs, warmups):
    result = {}
    for mode in _ARMS:
        samples = [row['phases_ns'] for row in runs if row['mode'] == mode and row['pair'] >= warmups]
        result[mode] = {}
        for phase in samples[0]:
            values = [sample[phase] for sample in samples]
            result[mode][phase] = dict(samples_ns=values, min_ns=min(values), median_ns=median(values), max_ns=max(values))
    return result


def _inputs(context, contract, paths):
    inputs = [_receipt(path) for path in paths]
    if [row['size_bytes'] for row in inputs] != [port['nbytes'] for port in contract['inputs']]:
        raise ValueError('lifecycle calibration inputs differ from the admitted source matrix ABI')
    if Path.cwd() != Path(context['working_directory']):
        raise ValueError('lifecycle calibration working directory changed')
    return inputs


def _commands(probe, context):
    return dict(ORIGINAL=[*context['prefix'], context['python']['path'], '-B', str(probe/'original.py')],
                NATIVE=[str(probe/'native/launch.sh')])


def _cohort(execution, target, original_cpu_ids):
    native = execution['cpu_ids']
    if original_cpu_ids is None:
        return native, None
    if (type(original_cpu_ids) is not list or any(type(cpu) is not int or cpu < 0 for cpu in original_cpu_ids)
            or original_cpu_ids != sorted(set(original_cpu_ids)) or not set(native) <= set(original_cpu_ids)):
        raise ValueError('the original lifecycle CPU scope must be canonical and include the native placement')
    from scar.analysis.target_graph import TargetHardwareGraph
    from scar.optimization.aten_native_schedule import recorded_cpu_domain

    graph = TargetHardwareGraph(Path(target['uri']).parent)
    _, domain = recorded_cpu_domain(graph)
    if not set(original_cpu_ids).issubset(domain):
        raise ValueError('the original lifecycle CPU scope is outside the retained hardware domain')
    return original_cpu_ids, dict(cpu_ids=native, threads=execution['threads'])


def _preparation(project, contract, manifest, probe, *, build):
    source = Path(contract['source']['path'])
    original, native = probe/'original.py', probe/'native'
    derived = {'CMakeLists.txt': (project/'CMakeLists.txt').read_bytes(),
        'include/scar_generated/matrix_plan.hpp': (project/'include/scar_generated/matrix_plan.hpp').read_bytes(),
        'launch.sh': (project/'launch.sh').read_bytes(),
        'src/main.cpp': _native_probe((project/'src/main.cpp').read_text()).encode()}
    python_bytes = _python_probe(source, contract).encode()
    if build:
        (native/'data').mkdir(parents=True)
        original.write_bytes(python_bytes)
        for name, raw in derived.items():
            path = native/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        (native/'launch.sh').chmod(0o755)
    if original.read_bytes() != python_bytes or any((native/name).read_bytes() != raw for name, raw in derived.items()):
        raise ValueError('lifecycle probes differ from the scoped Python and C++ instrumentation')
    _, stack = load_operator_stack(manifest['stack'])
    link = nvidia_stack_build(stack, 'CPU')
    if build:
        executable, dependencies = build_native_project(native, Path(manifest['executable']).name, link['compiler'],
                                                        expected_libraries=link['libraries'])
    else:
        executable = native/manifest['executable']
        dependencies = _dependencies(subprocess.check_output(['/usr/bin/ldd', str(executable)], env={}, text=True))
    linkage = verify_operator_stack_linkage(stack, dependencies, 'CPU')
    if linkage != manifest['linkage']:
        raise ValueError('instrumented NVIDIA probe uses a different numerical stack')
    return dict(source=_receipt(original), native_members={name: _receipt(native/name) for name in derived},
                binary=_receipt(executable), linkage=linkage)


def calibrate_nvhpc_source_lifecycle(project, manifest_sha256, arguments, out, *, warmups=1, repetitions=3,
                                     original_cpu_ids=None):
    """Measure the qualified file entry with a fixed original cohort if supplied."""
    import torch

    if (type(warmups) is not int or not 0 <= warmups <= 10
            or type(repetitions) is not int or not 3 <= repetitions <= 30):
        raise ValueError('lifecycle calibration needs a bounded warmup and at least three retained samples')
    root, project_ref, execution, recipe, _ = check_nvhpc_project(project, manifest_sha256)
    manifest = read_artifact(project_ref)
    from scar.analysis.nvhpc_source_entry import SCHEMA as SOURCE

    contract_ref = artifact_reference(root/'plan/source-entry.json', SOURCE)
    contract = read_artifact(contract_ref)
    target = json.loads((root/'plan/problem.json').read_bytes())['target']
    original_cpus, native_placement = _cohort(execution, target, original_cpu_ids)
    context = _context(original_cpus)
    paths = _calibration_paths(arguments, execution['inputs'])
    inputs = _inputs(context, contract, paths)
    sizes = [port['nbytes'] for port in contract['outputs']]
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    probe = out/'probe'
    preparation = _preparation(root, contract, manifest, probe, build=True)
    commands = _commands(probe, context)
    runs, comparisons = [], []
    with TemporaryDirectory(prefix='.lifecycle-observed-', dir=out) as temporary:
        scratch = Path(temporary)
        for pair in range(warmups + repetitions):
            content = {}
            for mode in (_ARMS if pair % 2 == 0 else _ARMS[::-1]):
                paths_out = [scratch/f'{index}.f32' for index in range(len(sizes))]
                argv = [*map(str, paths), *map(str, paths_out)]
                start = monotonic_ns()
                observed = subprocess.run([*commands[mode], *argv], env=context['environment'],
                                          capture_output=True, timeout=120, check=True)
                end = monotonic_ns()
                if observed.stdout or any(path.is_symlink() or not path.is_file() for path in paths_out):
                    raise ValueError('lifecycle instrumentation changed source stream or file effects')
                events = _parse_samples(observed.stderr)
                phases = _phase_times(start, end, events, _markers(contract, recipe, mode))
                payloads = [_content(path) for path in paths_out]
                if [row['size_bytes'] for row in payloads] != sizes:
                    raise ValueError('lifecycle probe publication differs from the admitted shape')
                content[mode] = [path.read_bytes() for path in paths_out]
                for path in paths_out:
                    path.unlink()
                runs.append(dict(pair=pair, mode=mode, start_ns=start, end_ns=end,
                    events=events, phases_ns=phases, output_payloads=payloads))
            quality = [_compare(left, right, execution['numerical_contract'], torch)
                       for left, right in zip(content['ORIGINAL'], content['NATIVE'], strict=True)]
            if not all(row['accepted'] for row in quality):
                raise ValueError('lifecycle probes did not preserve the declared FP32 output contract')
            comparisons.append(quality)

    if _inputs(context, contract, paths) != inputs:
        raise ValueError('source matrix inputs changed while observing lifecycle costs')
    check_nvhpc_project(root, manifest_sha256)
    if _context(original_cpus) != context or _preparation(root, contract, manifest, probe, build=False) != preparation:
        raise ValueError('lifecycle process, probe or numerical provider identity changed during collection')
    report = dict(schema=SCHEMA, schema_version=1, artifact=project_ref, source_contract=contract_ref,
        target=target, context=context, inputs=inputs, probe=preparation, commands=commands,
        comparison_scratch_device=out.stat().st_dev, scope='PROCESS_LAUNCH_THROUGH_EXIT',
        sample_kind='INSTRUMENTED_PHASE_OBSERVATIONS',
        cache_regime='FRESH_PROCESSES_REUSE_INPUT_FILES_NO_CACHE_FLUSH',
        limitations=_LIMITATIONS, warmups=warmups, repetitions=repetitions,
        comparisons=comparisons, runs=runs, summary=_summary(runs, warmups),
        model_bounds_proven=False, clean_ab=False, deployment_authorized=False)
    if native_placement is not None:
        report['native_placement'] = native_placement
    _write_record(out/'result.json', report)
    return report


def replay_nvhpc_source_lifecycle(reference):
    """Check probe identities and recompute the recorded component observations."""
    report = read_artifact(reference)
    artifact = report['artifact']
    root, _, execution, recipe, _ = check_nvhpc_project(Path(artifact['uri']).parent, artifact['sha256'])
    manifest = read_artifact(artifact)
    contract = read_artifact(report['source_contract'])
    path = Path(reference['uri']).parent
    target = json.loads((root/'plan/problem.json').read_bytes())['target']
    explicit = 'native_placement' in report
    original_cpus, native_placement = _cohort(execution, target, report['context']['cpu_ids'] if explicit else None)
    context = _context(original_cpus)
    if (reference['schema'] != SCHEMA or report['schema_version'] != 1
            or report['scope'] != 'PROCESS_LAUNCH_THROUGH_EXIT'
            or report['sample_kind'] != 'INSTRUMENTED_PHASE_OBSERVATIONS'
            or report['cache_regime'] != 'FRESH_PROCESSES_REUSE_INPUT_FILES_NO_CACHE_FLUSH'
            or report['limitations'] != _LIMITATIONS or report['context'] != context or report['target'] != target
            or explicit and report['native_placement'] != native_placement
            or report['commands'] != _commands(path/'probe', context)
            or report['comparison_scratch_device'] != path.stat().st_dev
            or report['source_contract'] != artifact_reference(root/'plan/source-entry.json', contract['schema'])
            or report['probe'] != _preparation(root, contract, manifest, path/'probe', build=False)
            or any(report.get(flag) is not False for flag in ('model_bounds_proven', 'clean_ab', 'deployment_authorized'))):
        raise ValueError('lifecycle evidence changed its source, provider, launch or measurement scope')
    if _inputs(context, contract, [item['path'] for item in report['inputs']]) != report['inputs']:
        raise ValueError('source matrix inputs differ from the lifecycle observation')
    warmups, repetitions = report['warmups'], report['repetitions']
    runs, comparisons = report['runs'], report['comparisons']
    sizes = [port['nbytes'] for port in contract['outputs']]
    if (type(warmups) is not int or not 0 <= warmups <= 10
            or type(repetitions) is not int or not 3 <= repetitions <= 30
            or len(runs) != 2 * (warmups + repetitions) or len(comparisons) * 2 != len(runs)
            or any(len(row) != len(sizes) or any(item['accepted'] is not True for item in row) for row in comparisons)):
        raise ValueError('retained lifecycle evidence has an incomplete sample design')
    for index, row in enumerate(runs):
        pair, slot = divmod(index, 2)
        mode = _ARMS[(pair + slot) % 2]
        if ((row['pair'], row['mode']) != (pair, mode)
                or [item['size_bytes'] for item in row['output_payloads']] != sizes
                or row['phases_ns'] != _phase_times(row['start_ns'], row['end_ns'], row['events'],
                                                    _markers(contract, recipe, mode))):
            raise ValueError('retained lifecycle timings differ from their exact source activity events')
        if slot:
            previous = runs[index-1]['output_payloads']
            for left, right, quality in zip(previous, row['output_payloads'], comparisons[pair], strict=True):
                if quality['exact_bytes'] is not (left['sha256'] == right['sha256']):
                    raise ValueError('lifecycle output comparison differs from its retained byte fingerprints')
    if report['summary'] != _summary(runs, warmups):
        raise ValueError('retained lifecycle phase statistics changed')
    read_artifact(reference, json_document=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    match = parser.add_mutually_exclusive_group(required=True)
    match.add_argument('--project')
    match.add_argument('--check')
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--input', action='append', default=[])
    parser.add_argument('--out')
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--original-cpus', type=int, nargs='+',
                        help='fixed original CPU cohort for comparing native placements within that cohort')
    args = parser.parse_args(argv)
    if args.check:
        if args.original_cpus is not None:
            parser.error('--check consumes the original CPU scope retained by calibration')
        report = replay_nvhpc_source_lifecycle(artifact_reference(args.check, SCHEMA))
    else:
        if not args.manifest_sha256 or not args.out:
            parser.error('--project requires --manifest-sha256 and --out')
        report = calibrate_nvhpc_source_lifecycle(args.project, args.manifest_sha256, args.input,
            args.out, warmups=args.warmups, repetitions=args.repetitions, original_cpu_ids=args.original_cpus)
    print(json.dumps(dict(sample_kind=report['sample_kind'], repetitions=report['repetitions'], summary=report['summary'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

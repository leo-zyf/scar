"""Diagnose one real guarded entry against an unchanged stateful ELF on Linux."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import ctypes
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import monotonic_ns
import uuid

from scar.compiler import stateful_holdout as sampler
from scar.compiler import stateful_lifetime_training as lifetime
from scar.ir.execution_recipe import artifact_reference, read_artifact

DESIGN = 'scar.stateful-entry-phase-design'
OBSERVATIONS = 'scar.stateful-entry-phase-observations'
NATIVE_APIS = ('workspace_new_array', 'workspace_delete_array', 'sched_getaffinity',
    'sched_setaffinity', 'sched_getcpu', 'affinity_mask_alloc', 'affinity_mask_free', 'sysconf')
_SOURCE = Path(__file__).resolve().parents[1]/'codegen/stateful_entry_phase_probe.cpp.in'


def _reference(path, schema):
    return artifact_reference(Path(path).resolve(), schema)


def _producer():
    return {name:_reference(path, 'scar.implementation-source')
        for name, path in (('runner', __file__), ('shim', _SOURCE))}


def prepare(lifetime_design, out):
    campaign_ref, campaign, template = lifetime._design(lifetime_design)
    manifest = read_artifact(template['native_artifact'])
    compiler = manifest['compiler']['argv']
    _need = lifetime._need
    _need(sys.platform.startswith('linux') and manifest['cpu_threads'] == 1
        and not os.environ.get('LD_PRELOAD'), 'one serial Linux process without an existing preload required')
    _need(subprocess.check_output([*compiler, '--version'], text=True).splitlines()[0]
        == manifest['compiler']['identity'], 'original compiler identity differs')
    directory = Path(out).resolve()
    directory.mkdir(parents=True)
    library = directory/'phase-probe.so'
    command = [*compiler, '-std=c++20', '-O2', '-shared', '-fPIC', '-x', 'c++',
        str(_SOURCE), '-x', 'none', '-o', str(library), '-Wl,--no-as-needed', '-lstdc++', '-ldl']
    subprocess.run(command, check=True)
    return sampler._write(directory/'design.json', dict(schema=DESIGN, schema_version=1,
        lifetime_design=campaign_ref, directory=str(directory), capture_epoch=uuid.uuid4().hex,
        original_native_manifest=template['native_artifact'], original_native_elf=template['native_members']['library'],
        original_protocol=campaign['context']['protocol'], producer=_producer(), compiler_command=command,
        shim=_reference(library, 'scar.diagnostic-native-shim'), warmups=2, repetitions=7,
        diagnostic_only=True, deployment_authorized=False))


def _load(path, digest):
    reference, design = lifetime._checked(path, DESIGN, digest)
    _, campaign, template = lifetime._design(design['lifetime_design'])
    lifetime._need(design['original_native_manifest'] == template['native_artifact']
        and design['original_native_elf'] == template['native_members']['library']
        and design['original_protocol'] == campaign['context']['protocol'] and design['producer'] == _producer()
        and (design['warmups'], design['repetitions']) == (2, 7), 'frozen diagnostic context differs')
    read_artifact(design['shim'], json_document=False)
    return reference, design, campaign, template


@contextmanager
def _cuts(guard, design):
    from scar.codegen.stateful import _RUN, _Table
    from scar.compiler.stateful_runtime import _ArtifactAdapter

    original = guard.scar_artifact
    adapter = inspect_adapter(guard)
    lifetime._need(type(adapter) is _ArtifactAdapter and adapter.original is original,
        'live shared guard uses a different adapter')
    library = ctypes.CDLL(design['shim']['uri'])
    bind = library.scar_phase_probe_bind_v1
    bind.argtypes, bind.restype = [_RUN], ctypes.c_char_p
    table_pointer = original._table
    origin = bind(table_pointer.contents.run)
    lifetime._need(origin and Path(os.fsdecode(origin)).resolve() == Path(design['original_native_elf']['uri']),
        'instrumented pointer does not resolve to the pinned original ELF')
    table = _Table.from_buffer_copy(table_pointer.contents)
    table.run = _RUN(('scar_phase_probe_run_v1', library))
    reader = library.scar_phase_probe_read_v1
    reader.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64]
    reader.restype = ctypes.c_int32
    words = (ctypes.c_uint64 * (2 + 3 * len(NATIVE_APIS)))()
    rows = []
    run = adapter.run
    adapter_cut = []

    def timed_adapter(arrays, scalars):
        enter = monotonic_ns()
        output = run(arrays, scalars)
        adapter_cut.append((enter, monotonic_ns()))
        return output

    def timed_guard(*args, **kwargs):
        adapter_cut.clear()
        enter = monotonic_ns()
        returned = guard(*args, **kwargs)
        leave = monotonic_ns()
        lifetime._need(len(adapter_cut) == 1 and reader(words, len(words)) == 0,
            'one successful original native call required for each diagnostic row')
        native_enter, native_exit = words[:2]
        apis = {key:dict(calls=words[2 + 3*i], elapsed_ns=words[3 + 3*i], requested_bytes=words[4 + 3*i])
            for i, key in enumerate(NATIVE_APIS)}
        rows.append(dict(cuts_ns=[enter, adapter_cut[0][0], native_enter, native_exit, adapter_cut[0][1], leave],
            direct_native_apis=apis))
        return returned

    original._table = ctypes.pointer(table)
    adapter.run = timed_adapter
    try:
        yield timed_guard, rows
    finally:
        del adapter.run
        original._table = table_pointer


def inspect_adapter(guard):
    from inspect import getclosurevars

    return getclosurevars(guard).nonlocals['artifact']


def _components(sample):
    cuts = sample['cuts_ns']
    lifetime._need(len(cuts) == 6 and all(type(t) is int and t > 0 for t in cuts)
        and cuts == sorted(cuts), 'all instrumented clocks must be on one ordered monotonic timeline')
    api = sample['direct_native_apis']
    lifetime._need(set(api) == set(NATIVE_APIS) and all(type(v) is int and v >= 0
        for row in api.values() for v in row.values()), 'complete directly observed API counters required')
    enter, adapter, native, native_end, adapter_end, end = cuts
    allocation = sum(api[key]['elapsed_ns'] for key in NATIVE_APIS[:2])
    affinity = sum(api[key]['elapsed_ns'] for key in NATIVE_APIS[2:])
    native_residual = native_end-native-allocation-affinity
    lifetime._need(native_residual >= 0 and api['sched_setaffinity']['calls'] > 0
        and api['workspace_new_array']['calls'] > 0, 'interposition must observe original affinity and allocation calls')
    return dict(guard_total_instrumented_ns=end-enter, native_abi_total_instrumented_ns=native_end-native,
        guard_before_adapter_ns=adapter-enter, guard_return_after_adapter_ns=end-adapter_end,
        adapter_outside_native_abi_ns=native-adapter+adapter_end-native_end,
        observed_native_workspace_api_ns=allocation, observed_native_affinity_api_ns=affinity,
        native_residual_ns=native_residual)


def _distribution(samples):
    return {name:dict(median_ns=median(sample[name] for sample in samples),
        min_ns=min(sample[name] for sample in samples), max_ns=max(sample[name] for sample in samples))
        for name in samples[0]}


def _statistics(episodes, design, campaign, workspace):
    lifetime._need(len(episodes) == design['warmups']+design['repetitions'], 'all original diagnostic episodes required')
    first, later = [], []
    counts = []
    for index, episode in enumerate(episodes):
        lifetime._need(episode['index'] == index and episode['oracle_equal'] is True,
            'all diagnostic episodes must match the unarmed original ELF')
        comps = [_components(sample) for sample in episode['samples']]
        lifetime._need(len(comps) == campaign['method_calls_per_episode'], 'declared original call trajectory differs')
        for sample in episode['samples']:
            a = sample['direct_native_apis']
            lifetime._need(a['workspace_new_array']['requested_bytes'] == workspace
                and a['workspace_new_array']['calls'] == a['workspace_delete_array']['calls'],
                'observed original array payload requests and release counts differ')
        if index >= design['warmups']:
            first.append(comps[0])
            later.extend(comps[1:])
            counts.extend({name:row['calls'] for name, row in sample['direct_native_apis'].items()}
                for sample in episode['samples'])
    return dict(first=_distribution(first), later=_distribution(later),
        api_call_count_ranges={name:[min(row[name] for row in counts), max(row[name] for row in counts)]
            for name in NATIVE_APIS})


def _run(path, digest):
    ref, design, campaign, template = _load(path, digest)
    lifetime._need(os.environ.get('LD_PRELOAD') == design['shim']['uri'], 'lifetime diagnostic must own its preload scope')
    directory = Path(design['directory'])
    recorder = deepcopy(template)
    recorder['request']['capture_epoch'] = design['capture_epoch']
    workspace = read_artifact(template['native_artifact'])['workspace_payload_bytes']
    from scar.runtime.compiled_deployment import thread_environment

    def capture(spec, func):
        episode = sampler._episode(template['request']['entry']['uri'], module, region, signature,
            func, guard, recorder, spec)
        lifetime._validate_steps(campaign, 'B', design['capture_epoch'], episode)
        lifetime._need(episode['execution']['exit_code'] == 0, 'diagnostic original driver did not complete')
        return episode

    scope = {**campaign, 'directory':str(directory)}
    episodes = []
    with lifetime._runtime(campaign, template) as (module, region, signature, _, guard):
        oracle = capture(lifetime._planned(scope, '.', 'oracle', 0, 'B'), guard)
        with _cuts(guard, design) as (observe, rows):
            for index in range(design['warmups']+design['repetitions']):
                rows.clear()
                record = capture(lifetime._planned(scope, '.', 'diagnostic', index, 'B'), observe)
                equal = all(record[key] == oracle[key] for key in ('final', 'stdout', 'stderr', 'files'))
                episodes.append(dict(index=index, oracle_equal=equal, samples=list(rows)))
                lifetime._need(equal, 'diagnostic and unarmed original ELF outputs differ')
    lifetime._need(thread_environment() == campaign['context']['native_thread_environment'],
        'native thread environment differed after diagnostic')
    for item in (design['original_native_elf'], template['request']['entry'], *template['source_refs'].values()):
        read_artifact(item, json_document=False)
    summary = _statistics(episodes, design, campaign, workspace)
    return sampler._write(directory/'observations.json', dict(schema=OBSERVATIONS, schema_version=1,
        design=ref, original_elf_path_executed=True, original_elf_recompiled=False,
        original_elf_pointer_replaced_by_measurement_shim_in_private_table_copy=True,
        dynamic_symbols_direct_origin='CALL_SITE_IN_ORIGINAL_EXECUTABLE_ELF_SEGMENT',
        software_clock='MONOTONIC_NS_PYTHON_AND_CLOCK_MONOTONIC_CPP',
        scope='INSTRUMENTED_REAL_GUARDED_METHOD_START_THROUGH_RETURN',
        episodes=episodes, summary=summary, workspace_api_requested_bytes=workspace,
        native_residual_contains=['inline guards and finite scans', 'kernel loop bodies and provider calls',
            'floating-point scopes', 'commit copies', 'unmeasured cleanup', 'clock/interposition overhead'],
        allocation_api_implies_first_touch=False, optimizer_cost_qualification=False,
        clean_performance_claim=False, deployment_authorized=False))


def check(observations, digest):
    _, observed = lifetime._checked(observations, OBSERVATIONS, digest)
    _, design, campaign, template = _load(observed['design'], observed['design']['sha256'])
    workspace = read_artifact(template['native_artifact'])['workspace_payload_bytes']
    lifetime._need(observed['summary'] == _statistics(observed['episodes'], design, campaign, workspace),
        'diagnostic summary differs from the original raw cuts')
    return observed['summary']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('prepare')
    freeze.add_argument('--lifetime-design', required=True)
    freeze.add_argument('--out', required=True)
    for command, key in (('run', 'design'), ('capture', 'design'), ('check', 'observations')):
        sub = commands.add_parser(command)
        sub.add_argument('--'+key, required=True)
        sub.add_argument('--sha256', required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        output = prepare(args.lifetime_design, args.out)
    elif args.command == 'run':
        _, design = lifetime._checked(args.design, DESIGN, args.sha256)
        lifetime._need('LD_PRELOAD' not in os.environ, 'diagnostic must own subprocess preload')
        return subprocess.run([sys.executable, '-m', __spec__.name, 'capture', '--design', args.design,
            '--sha256', args.sha256], env={**os.environ, 'LD_PRELOAD':design['shim']['uri']}).returncode
    elif args.command == 'capture':
        output = _run(args.design, args.sha256)
    else:
        output = check(args.observations, args.sha256)
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

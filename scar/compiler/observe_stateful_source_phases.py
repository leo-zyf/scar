"""Replay source anchors and diagnose the original serial stateful frame."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import ctypes
from hashlib import sha256
import json
from math import prod
from pathlib import Path
from statistics import median
import subprocess
from time import monotonic_ns
import uuid

from scar.codegen.native_stage_group import render_native_stage_group
from scar.compiler import stateful_holdout as sampler
from scar.compiler import stateful_lifetime_training as lifetime
from scar.ir.execution_recipe import artifact_reference, read_artifact

DESIGN = 'scar.stateful-source-phase-design'
OBSERVATIONS = 'scar.stateful-source-phase-observations'
PHASES = ('finite_input_scan', 'snapshot', 'computation_with_provider', 'commit')
_CUT = 'scar_source_phase_cut_v1'
_DECLARATIONS = '''
namespace {
thread_local uint64_t scar_source_phase_cuts_v1[6] = {};
__attribute__((noinline)) void scar_source_phase_cut_v1(size_t index) {
    asm volatile("" ::: "memory");
    timespec stamp{};
    scar_source_phase_cuts_v1[index] =
        clock_gettime(CLOCK_MONOTONIC, &stamp) == 0
        ? uint64_t(stamp.tv_sec) * 1000000000ULL + uint64_t(stamp.tv_nsec) : 0;
    asm volatile("" ::: "memory");
}
}
'''
_EXPORTS = '''
extern "C" SCAR_EXPORT_V1 int32_t scar_source_phase_run_v1(
        const scar_array_v1 *arrays, uint64_t array_count,
        const scar_scalar_v1 *scalars, uint64_t scalar_count,
        const scar_provider_v1 *providers, uint64_t provider_count) {
    for (auto &word : scar_source_phase_cuts_v1) word = 0;
    scar_source_phase_cut_v1(0);
    int32_t status = scar_get_stateful_table_v1()->run(
        arrays, array_count, scalars, scalar_count, providers, provider_count);
    scar_source_phase_cut_v1(5);
    return status;
}
extern "C" SCAR_EXPORT_V1 int32_t scar_source_phase_read_v1(uint64_t *out, uint64_t count) {
    if (!out || count != 6) return 1;
    for (size_t i = 0; i < count; ++i) {
        if (!scar_source_phase_cuts_v1[i]) return 1;
        out[i] = scar_source_phase_cuts_v1[i];
        scar_source_phase_cuts_v1[i] = 0;
    }
    return 0;
}
'''


def _sha(text):
    return sha256(text.encode()).hexdigest()


def _anchor(source, block):
    lifetime._need(block and source.count(block) == 1, 'one exact generated block per phase required')
    begin = source.index(block)
    return begin, begin + len(block)


def _source(manifest, plan_ref, region_ref):
    plan, region = map(read_artifact, (plan_ref, region_ref))
    stages = [row for row in plan['stages'] if row['kind'] != 'COMMIT']
    snapshot = [row['index'] for row in stages if row['kind'] == 'SNAPSHOT_ROOT']
    computation = [row['index'] for row in stages if row['kind'] != 'SNAPSHOT_ROOT']
    lifetime._need(plan['cpu_threads'] == 1 and snapshot and computation
        and snapshot + computation == [row['index'] for row in stages]
        and any(row['kind'] == 'PROVIDER' for row in stages), 'serial snapshot then provider/computation stages required')
    grouped, provenance = render_native_stage_group(manifest, groups=[snapshot, computation])
    lifetime._need(provenance['parent_view']['plan'] == plan_ref
        and provenance['parent_view']['region'] == region_ref
        and _CUT not in grouped and provenance['cut_count'] == 4, 'observer is not the original source plan')
    original = read_artifact(provenance['parent_view']['source'], json_document=False).decode()
    arrays = [row['id'] for row in region['inputs'] if row['kind'] == 'array']
    scans = '\n'.join(
        f'    for (int64_t i = 0; i < {prod(plan["input_shapes"][name])}; ++i) '
        f'if (!finite(static_cast<const double *>(arrays[{i}].data)[i])) return 3;'
        for i, name in enumerate(arrays))
    slots = {row['id']:index for index, row in enumerate(plan['workspace_slots'])}
    commits = []
    for row in plan['commit_rows']:
        target = row['target']
        (r, re), (c, ce) = row['normalized_bounds']
        stride = plan['input_shapes'][target][1]
        slot = slots[plan['assignment'][plan['shadows'][target]]['slot_id']]
        commits.append(f'    for (int64_t row = {r}; row < {re}; ++row) if ({ce-c})\n'
            f'        std::memcpy(static_cast<double *>(arrays[{arrays.index(target)}].data) + row * {stride} + {c}, '
            f'slots[{slot}].get() + row * {stride} + {c}, {ce-c} * sizeof(double));')
    additions = [(grouped.index('namespace {'), _DECLARATIONS), (len(grouped), _EXPORTS)]
    anchors = {}
    for phase, block, first_cut in ((PHASES[0], scans, 1), (PHASES[3], '\n'.join(commits), 3)):
        a, b = _anchor(original, block)
        start, end = _anchor(grouped, block)
        anchors[phase] = dict(original_start=a, original_end=b, block_sha256=_sha(block),
            anchor_authority='PLAN_RECONSTRUCTS_UNIQUE_BLOCK_IN_ORIGINAL_CPP')
        additions.extend(((start, f'    {_CUT}({first_cut});\n'), (end, f'\n    {_CUT}({first_cut+1});')))
    observed = grouped
    for offset, text in sorted(additions, reverse=True):
        observed = observed[:offset] + text + observed[offset:]
    shift = 0
    insertions = []
    for offset, text in sorted(additions):
        insertions.append(dict(start=offset+shift, end=offset+shift+len(text), sha256=_sha(text)))
        shift += len(text)
    restored = observed
    for row in reversed(insertions):
        restored = restored[:row['start']] + restored[row['end']:]
    lifetime._need(restored == grouped and provenance['all_original_source_bytes_preserved'],
        'all original source bytes must survive insertion')
    return observed, dict(observer=provenance, unique_original_anchors=anchors, insertions=insertions)


def _derive(campaign_path):
    campaign_ref, campaign, template = lifetime._design(campaign_path)
    members = template['native_members']
    source, provenance = _source(template['native_artifact'],
        _typed(members['plan'], 'scar.stateful-stage-plan'), _typed(members['region'], 'scar.stateful-array-region'))
    return campaign_ref, campaign, template, source, provenance


def _typed(reference, schema):
    read_artifact(reference, json_document=False)
    return {**reference, 'schema':schema}


def prepare(campaign_path, out):
    campaign_ref, _, template, source, phases = _derive(campaign_path)
    directory = Path(out).resolve()
    directory.mkdir(parents=True)
    manifest = read_artifact(template['native_artifact'])
    compiler = manifest['compiler']
    lifetime._need(subprocess.check_output([*compiler['argv'], '--version'], text=True).splitlines()[0]
        == compiler['identity'], 'original compiler identity differs')
    view = phases['observer']['parent_view']
    headers = {}
    for name, ref in view['headers'].items():
        path = directory/name
        path.write_bytes(read_artifact(ref, json_document=False))
        headers[name] = artifact_reference(path, ref['schema'])
    src, library = directory/'observer.cpp', directory/'observer.so'
    src.write_text(source)
    command = [*compiler['argv'], *compiler['options'], str(src), '-o', str(library), '-lm']
    subprocess.run(command, check=True)
    return sampler._write(directory/'design.json', dict(schema=DESIGN, schema_version=1,
        lifetime_design=campaign_ref, directory=str(directory), capture_epoch=uuid.uuid4().hex,
        phases=phases, source=artifact_reference(src, 'scar.cpp-source'), headers=headers,
        library=artifact_reference(library, 'scar.diagnostic-native-library'), compiler_command=command,
        implementation=artifact_reference(Path(__file__), 'scar.implementation-source'),
        warmups=2, repetitions=7, deployment_authorized=False))


def _load(value, digest):
    ref, design = lifetime._checked(value, DESIGN, digest)
    _, campaign, template, source, phases = _derive(design['lifetime_design'])
    lifetime._need(design['phases'] == phases and read_artifact(design['source'], json_document=False) == source.encode()
        and design['implementation'] == artifact_reference(Path(__file__), 'scar.implementation-source')
        and (design['warmups'], design['repetitions']) == (2, 7), 'source phase design differs from cold replay')
    for name, ref_header in design['headers'].items():
        lifetime._need(read_artifact(ref_header, json_document=False) ==
            read_artifact(phases['observer']['parent_view']['headers'][name], json_document=False), 'ABI header differs')
    read_artifact(design['library'], json_document=False)
    return ref, design, campaign, template


@contextmanager
def _cuts(guard, design):
    from scar.codegen.stateful import _RUN, _Table

    artifact = guard.scar_artifact
    original = artifact._table
    library = ctypes.CDLL(design['library']['uri'])
    getter = library.scar_get_stateful_table_v1
    getter.argtypes, getter.restype = [], ctypes.POINTER(_Table)
    phase_table = getter().contents
    identity_fields = ('abi_version', 'struct_size', 'array_count', 'scalar_count', 'provider_count',
        'region_sha256', 'plan_sha256', 'contract_sha256')
    lifetime._need(all(getattr(phase_table, field) == getattr(original.contents, field)
        for field in identity_fields), 'instrumented frame uses a different original ABI contract')
    provenance = design['phases']['observer']
    identity = getattr(library, provenance['identity_symbol'])
    identity.argtypes, identity.restype = [], ctypes.c_char_p
    lifetime._need(identity() == provenance['observer_identity'].encode(), 'wrong original stage-group source identity')
    readers, data = [], []
    for symbol, count in ((provenance['read_symbol'], 4), ('scar_source_phase_read_v1', 6)):
        reader = getattr(library, symbol)
        reader.argtypes, reader.restype = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64], ctypes.c_int32
        readers.append(reader)
        data.append((ctypes.c_uint64 * count)())
    table = _Table.from_buffer_copy(original.contents)
    table.run = _RUN(('scar_source_phase_run_v1', library))
    rows = []

    def observe(*args, **kwargs):
        start = monotonic_ns()
        output = guard(*args, **kwargs)
        end = monotonic_ns()
        lifetime._need(all(reader(buffer, len(buffer)) == 0 for reader, buffer in zip(readers, data)),
            'all observer cuts must follow one successful native call')
        groups, extra = data
        rows.append([start, *extra[:3], *groups, *extra[3:], end])
        return output

    artifact._table = ctypes.pointer(table)
    try:
        yield observe, rows
    finally:
        artifact._table = original


def _components(cuts):
    lifetime._need(len(cuts) == 12 and cuts == sorted(cuts) and all(type(n) is int and n > 0 for n in cuts),
        'one monotone integer clock timeline with all original cuts required')
    spans = {phase:cuts[hi] - cuts[lo] for phase, lo, hi in zip(PHASES, (2, 4, 6, 8), (3, 5, 7, 9))}
    native = cuts[-2]-cuts[1]
    return dict(guard=cuts[-1]-cuts[0], native_abi=native,
        native_outside_observed_phases=native-sum(spans.values()), **spans)


def _statistics(episodes, design, campaign):
    warmup = design['warmups']
    lifetime._need(len(episodes) == warmup + design['repetitions'], 'complete planned episodes required')
    first, later, deltas = [], [], []
    for index, episode in enumerate(episodes):
        lifetime._need(episode['index'] == index and episode['unmodified_elf_oracle_equal'] is True
            and len(episode['raw_cuts_ns']) == campaign['method_calls_per_episode'],
            'exact original method trajectory and original-ELF correctness required')
        before, *after = [_components(cuts) for cuts in episode['raw_cuts_ns']]
        if index >= warmup:
            first.append(before)
            later.extend(after)
            deltas.append({key:value-median(row[key] for row in after) for key, value in before.items()})

    def distribution(rows):
        return {key:dict(median_ns=median(row[key] for row in rows), min_ns=min(row[key] for row in rows),
            max_ns=max(row[key] for row in rows), positive_samples=sum(row[key] > 0 for row in rows))
            for key in rows[0]}

    return dict(first=distribution(first), continuation=distribution(later),
        paired_delta_first_minus_episode_continuation_median=distribution(deltas))


def run(value, digest):
    ref, design, campaign, template = _load(value, digest)
    directory = Path(design['directory'])
    sampler_context = deepcopy(template)
    sampler_context['request']['capture_epoch'] = design['capture_epoch']

    def capture(phase, index, func):
        planned = lifetime._planned({**campaign, 'directory':str(directory)}, '.', phase, index, 'B')
        episode = sampler._episode(campaign['context']['entry']['uri'], module, region, signature,
            func, guard, sampler_context, planned)
        lifetime._validate_steps(campaign, 'B', design['capture_epoch'], episode)
        lifetime._need(episode['execution']['exit_code'] == 0, 'original driver must complete')
        return episode

    episodes = []
    with lifetime._runtime(campaign, template) as (module, region, signature, _, guard):
        oracle = capture('oracle', 0, guard)
        with _cuts(guard, design) as (observe, cuts):
            for index in range(design['warmups'] + design['repetitions']):
                cuts.clear()
                output = capture('phase', index, observe)
                equal = all(output[key] == oracle[key] for key in ('final', 'stdout', 'stderr', 'files'))
                episodes.append(dict(index=index, raw_cuts_ns=list(cuts), unmodified_elf_oracle_equal=equal))
                lifetime._need(equal, 'instrumented original frame differs from the unmodified original ELF')
    for item in (template['native_members']['library'], template['request']['entry'], *template['source_refs'].values()):
        read_artifact(item, json_document=False)
    summary = _statistics(episodes, design, campaign)
    return sampler._write(directory/'observations.json', dict(schema=OBSERVATIONS, schema_version=1,
        design=ref, episodes=episodes, summary=summary,
        raw_cut_order=['guard_before', 'native_abi_before', 'finite_before', 'finite_after',
            'snapshot_before', 'snapshot_after', 'computation_before', 'computation_after',
            'commit_before', 'commit_after', 'native_abi_after', 'guard_after'],
        outer_scope='RECOMPILED_DIAGNOSTIC_FRAME_WITH_SAME_ORIGINAL_CPU_AFFINITY_AND_PROVIDER',
        production_elf_unchanged=True, observer_is_original_machine_code=False,
        each_phase_uses_same_run_monotonic_cut_pair=True, component_medians_summed=False,
        potential_probe_perturbations=['CLOCK_AND_TLS_WRITES', 'REGISTER_PRESSURE_AND_OPTIMIZATION',
            'UNCONTROLLED_FIRST_TOUCH_CACHE_NUMA_AND_COSCHEDULED_WORK'],
        clean_performance_claim=False, qualified_cost=False, deployment_authorized=False))


def check(value, digest):
    _, observed = lifetime._checked(value, OBSERVATIONS, digest)
    _, design, campaign, _ = _load(observed['design'], observed['design']['sha256'])
    expected = _statistics(observed['episodes'], design, campaign)
    lifetime._need(expected == observed['summary'], 'summary must replay every original capture')
    return expected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    freeze = subparsers.add_parser('prepare')
    freeze.add_argument('--lifetime-design', required=True)
    freeze.add_argument('--out', required=True)
    for action, key in (('run', 'design'), ('check', 'observations')):
        command = subparsers.add_parser(action)
        command.add_argument('--'+key, required=True)
        command.add_argument('--sha256', required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare(args.lifetime_design, args.out)
    elif args.command == 'run':
        result = run(args.design, args.sha256)
    else:
        result = check(args.observations, args.sha256)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

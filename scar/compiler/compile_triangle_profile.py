"""Plan, lower and validate an admitted fixed-count source triangle profile."""

import argparse
from hashlib import file_digest, sha256
import json
from math import isfinite
import os
from pathlib import Path
from shutil import copytree
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns, time_ns

from scar.codegen import native_parameterized_upstream as upstream
from scar.codegen import native_publication_delivery
from scar.codegen import native_source_triangle_profile as delivery
from scar.codegen.native_ordered_profile import export_profile_solver
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import native_ordered_profile as profiles
from scar.optimization.native_ordered_publication import ENV

SCHEMA = 'scar.source-triangle-profile-compilation'
BENCHMARK = 'scar.source-triangle-profile-benchmark'
AUTOMATIC = 'scar.source-triangle-entry-compilation'


def _require(valid, reason):
    if not valid:
        raise ValueError(reason)


def _write(path, record):
    with path.open('x', encoding='utf-8') as output:
        json.dump(record, output, indent=2, sort_keys=True, allow_nan=False)
        output.write('\n')
    return artifact_reference(path, record['schema'])


def _hash(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def _launch_environment(context):
    return {**ENV, **{key: value for key, value in context['environment'].items() if value is not None}}


def _stage_measurement(root, packages):
    """Hold executable and library identity constant wherever arms agree."""
    staged = {}
    for width, reference in packages.items():
        stage = root / width
        copytree(Path(reference['uri']).parent, stage, copy_function=os.link)
        staged[width] = stage
    first, *others = staged.values()
    if not others:
        return staged, []
    shared = []
    for member in first.rglob('*'):
        if not member.is_file() or not (os.access(member, os.X_OK) or '.so' in member.name):
            continue
        relative = member.relative_to(first)
        peers = [folder/relative for folder in others]
        if not all(peer.is_file() and peer.stat().st_size == member.stat().st_size for peer in peers):
            continue
        digest = _hash(member)
        if all(peer.samefile(member) or _hash(peer) == digest for peer in peers):
            for peer in peers:
                if not peer.samefile(member):
                    peer.unlink()
                    peer.hardlink_to(member)
            shared.append(str(relative))
    return staged, sorted(shared)


def _measure(out, plans, packages, *, repeats):
    archive = [read_artifact(reference) for reference in plans.values()]
    first = archive[0]
    source = read_artifact(first['source_seed'])
    release = read_artifact(first['anchor']['release'])
    context = release['context']
    expected = read_artifact(first['oracle'], json_document=False)
    for profile in archive:
        _require(profile['source_seed'] == first['source_seed'] and profile['oracle'] == first['oracle']
            and profile['anchor']['release'] == first['anchor']['release'],
            'schedule candidates must share the exact source boundary and oracle')
    cpus = sorted({cpu for profile in archive for cpu in profile['resources']['placement']['launch_cpu_ids']})
    _require(set(cpus) <= os.sched_getaffinity(0), 'benchmark requires the complete planned CPU cohort')
    argv = source['argv']
    cwd = context['cwd']
    environment = _launch_environment(context)
    interpreter = context['interpreter']['executable']
    entry = source['entry']
    prefix = ['/usr/bin/taskset', '-c', ','.join(map(str, cpus))]
    raw = []
    with TemporaryDirectory(prefix='.scar-profile-timing-', dir=out) as temporary:
        staged, shared = _stage_measurement(Path(temporary), packages)
        commands = {'original': [*prefix, interpreter, '-B', entry['uri'], *argv]}
        commands.update({name: [*prefix, str(stage/'scar_publish'), *argv] for name,stage in staged.items()})
        cases = list(commands)
        for index in range(repeats + 1):
            offset = index % len(cases)
            for case in cases[offset:] + cases[:offset]:
                begin = perf_counter_ns()
                result = subprocess.run(commands[case], stdin=subprocess.DEVNULL, capture_output=True,
                                        env=environment, cwd=cwd, timeout=125, check=True)
                elapsed = perf_counter_ns() - begin
                _require(result.stdout == expected and not result.stderr,
                    'a fresh launch diverged from the original publication in arm ' + case)
                raw.append(dict(round=index, case=case, stdout_sha256=sha256(result.stdout).hexdigest(),
                    stderr_size=len(result.stderr), elapsed_ns=elapsed))
    arms = {}
    for case in cases:
        samples = [row['elapsed_ns'] for row in raw if row['round'] > 0 and row['case'] == case]
        arms[case] = dict(samples_ns=samples, min_ns=min(samples), max_ns=max(samples), median_ns=median(samples))
    launch = dict(cpu_ids=cpus, argv=argv, environment=environment, entry=entry, cwd=cwd,
        interpreter=interpreter, stdin='DEVNULL', internal_native_cwd='/',
        shared_executable_or_library_paths=shared)
    return dict(schema=BENCHMARK, schema_version=1, plans=plans, packages=packages,
        launch=launch, raw=raw, arms=arms, warmup_rounds=1, measured_rounds=repeats)


def _model_input_matches(canonical, model):
    matched = model['matched_input']
    return all(matched[key] == canonical[key] for key in ('sha256', 'size_bytes'))


def _nominate_width(out, pilot, solver, common, oracle, admissible):
    from scar.codegen.native_ordered_timing import collect
    from scar.optimization.native_ordered_width_cost import prepare_width_proposal, project_transport

    canonical = read_artifact(common)['canonical_csr']
    input_path = Path(common['uri']).parent / canonical['uri']
    diagnostic = collect(solver, out/'calibration', input_path=input_path,
                         expected_stdout=oracle)
    replay = prepare_width_proposal(pilot, (diagnostic,))
    proposal = _write(out/'width-cost.json', replay.proposal)
    model = project_transport(proposal, replay)
    width = model['recommended_width']
    _require(type(width) is int and width in admissible and model['admissible_widths'] == list(admissible)
        and model['profile_plan'] == pilot and model['native_observations'] == [diagnostic]
        and model['pilot_proposal'] == proposal and model['holdout_used_for_nomination'] is False
        and _model_input_matches(canonical, model),
        'native width model differs from its exact pilot, evidence or allowed cohort')
    return width, dict(pilot_width=admissible[0], observation=diagnostic, proposal=proposal,
                      transport_proposal=_write(out/'transport-cost.json', model))


def compile_profile(anchor, source_seed, hardware, oracle, observation, out, *, widths, repeats=3,
                    calibrate_widths=False):
    """Reuse the checked upstream provider family and measure fresh scoped schedules."""
    anchor = Path(anchor).resolve(strict=True)
    baseline = native_publication_delivery.check(anchor)
    _require(baseline['schema_version'] == 6 and type(repeats) is int and repeats >= 3
        and len(widths) >= 1 and len(widths) == len(set(widths)),
        'supply a source-sealed provider family and distinct measured widths')
    out = Path(out).resolve()
    _require(not out.is_relative_to(anchor), 'profile output must be outside the provider family')
    out.mkdir(parents=True, exist_ok=False)
    admissible = tuple(sorted(widths))
    use_model = calibrate_widths and len(admissible) > 1
    plans = {}

    def prepare(width):
        schedule = profiles.plan_profile(anchor/'digest', source_seed, hardware, oracle, width,
            original_observation_path=observation)
        reference = _write(out/f'width-{width}.json', schedule)
        export_profile_solver(reference['uri'], out/f'solver-{width}')
        plans[str(width)] = reference
        return schedule

    pilot = admissible[0]
    first = prepare(pilot)
    resources = first['resources']
    identity = resources['recipe_id']
    common = upstream.export(anchor, source_seed, count=resources['case']['call_count'],
        recipe_id=identity, out=out/'upstream')
    width_model = None
    additional = admissible[1:]
    if use_model:
        width, width_model = _nominate_width(out, plans[str(pilot)], out/f'solver-{pilot}', common,
                                             oracle, admissible)
        additional = (width,) if width != pilot else ()
    for width in additional:
        schedule = prepare(width)
        _require(schedule['resources']['recipe_id'] == identity,
            'schedule widths must share one numerical wire identity')
    packages = {width: delivery.export(ref['uri'], out/'upstream', out/f'solver-{width}', out/f'native-{width}')
                for width, ref in plans.items()}
    benchmark = _write(out/'benchmark.json', _measure(out, plans, packages, repeats=repeats))
    selection = profiles.select_profile(benchmark['uri'])
    decision = _write(out/'selection.json', selection)
    result = dict(schema=SCHEMA, schema_version=1, source_seed=first['source_seed'],
        original_observation=first['original_observation'], oracle=first['oracle'],
        upstream=common, plans=plans, packages=packages, benchmark=benchmark, decision=decision,
        selected=selection['recommended_arm'], selected_native_width=selection['selected_width'],
        full_original_effects_closed=False)
    if width_model is not None:
        result['width_model'] = width_model
    return _write(out/'result.json', result)


def _check_acquisition(acquired, entry, argv):
    seed = read_artifact(acquired['source_seed'])
    observed = read_artifact(acquired['original_observation'])
    oracle = read_artifact(acquired['oracle'], json_document=False)
    frame = read_artifact(seed['provenance']['frames'])
    frontend = read_artifact(acquired['frontend'])
    _require(frontend == {key: value for key, value in acquired.items() if key != 'frontend'}
        and frontend['original_processes'] == 1 and frontend['prior_runtime_observations_transferred'] is False
        and seed['entry'] == observed['entry'] == entry and seed['argv'] == observed['argv'] == argv
        and frame['process_id'] == observed['process_id'] and frame['thread_id'] == observed['thread_id']
        and frame['cpu_ids'] == observed['affinity']
        and oracle == (observed['return_digest'] + '\n').encode('ascii')
        and sha256(oracle).hexdigest() == observed['expected_stdout_sha256'],
        'call bindings, loop completion and stdout must come from the same original child')


def compile_entry(family_path, entry, argv, out, *, widths=None, repeats=3, max_capture_seconds=60,
                  resume=False):
    """Collect the fresh evidence required by a registered native provider family."""
    from scar.compiler.observe_triangle_profile import observe_profile
    from scar.compiler.triangle_profile_family import SCHEMA as FAMILY, resolve
    from scar.runtime.sparse_ordered_scope_observe import scope_design

    _require(type(argv) in (list, tuple) and len(argv) == 1 and type(argv[0]) is str
        and argv[0].isascii() and argv[0].isdecimal() and 0 < len(argv[0]) <= 9,
        'expected one canonical finite iteration argument')
    count = int(argv[0])
    _require(count > 1 and argv[0] == str(count), 'expected more than one iteration without leading zeros')
    _require(isfinite(max_capture_seconds) and max_capture_seconds > 0, 'expected a positive source capture budget')
    argv = list(argv)
    entry = Path(entry).resolve(strict=True)
    family_ref = artifact_reference(Path(family_path).resolve(strict=True), FAMILY)
    selected = resolve(family_ref['uri'], entry, interpreter=sys.executable)
    read_artifact(family_ref)
    anchor = Path(selected['anchor']['uri']).parent
    entry_ref = artifact_reference(entry, 'scar.python-source')
    _require(entry_ref == selected['entry'], 'entry changed during provider admission')
    context = selected['context']
    out = Path(out).resolve()
    _require(not out.is_relative_to(anchor), 'profile output must be outside the provider family')
    available, admissible = profiles.observe_profile_host(anchor/'digest', count, widths=widths)
    aggregate = scope_design(anchor/'digest', argv)
    if resume:
        _require((out/'frontend'/'observed'/'result.json').is_file()
            and not (out/'compiled').exists() and not (out/'result.json').exists(),
            'resumption needs an existing source capture and a fresh downstream compilation')
    else:
        out.mkdir(parents=True, exist_ok=False)
    hardware_path = out/(f'hardware-{time_ns()}.json' if resume else 'hardware.json')
    hardware = _write(hardware_path, available)
    acquired = observe_profile(selected['source_seed']['uri'], argv, out/'frontend',
        aggregate_design=aggregate, interpreter=context['interpreter'],
        cwd=context['cwd'], environment=_launch_environment(context), expected_entry=entry,
        max_seconds=max_capture_seconds, cpu_ids=selected['launch_cpu_ids'][:1], resume=resume)
    _check_acquisition(acquired, entry_ref, argv)
    compiled = compile_profile(anchor, acquired['source_seed']['uri'], hardware['uri'],
        acquired['oracle']['uri'], acquired['original_observation']['uri'], out/'compiled',
        widths=admissible, repeats=repeats, calibrate_widths=widths is None)
    measured_widths = sorted(map(int, read_artifact(compiled)['plans']))
    return _write(out/'result.json', dict(schema=AUTOMATIC, schema_version=1, family=family_ref,
        entry=entry_ref, argv=argv, anchor=selected['anchor'], acquisition=acquired,
        hardware=hardware, candidate_widths=measured_widths, compilation=compiled, resumed_capture=resume))


def _check_width_model(record, plans, canonical):
    if 'transport_proposal' in record:
        from scar.optimization.native_ordered_width_cost import check_transport

        reference = record['transport_proposal']
        model = check_transport(reference['uri'])
        _require(model['pilot_proposal'] == record['proposal'] and model['holdout_used_for_nomination'] is False,
            'transport proposal differs from its calibrated pilot')
    else:
        from scar.optimization.native_ordered_width_cost import check as verify_width_cost

        reference = record['proposal']
        model = verify_width_cost(reference['uri'])
    pilot = record['pilot_width']
    _require(model == read_artifact(reference)
        and pilot == model['admissible_widths'][0]
        and model['profile_plan'] == plans[str(pilot)]
        and model['native_observations'] == [record['observation']]
        and _model_input_matches(canonical, model)
        and set(plans) == {str(pilot), str(model['recommended_width'])},
        'clean benchmark differs from the replayed native width nomination')


def check(path):
    """Replay the prepared artifacts and selection without timing the workload again."""
    reference = artifact_reference(Path(path).resolve(strict=True), SCHEMA)
    saved = read_artifact(reference)
    benchmark = read_artifact(saved['benchmark'])
    selection = profiles.verify_profile_selection(saved['decision']['uri'])
    common = saved['upstream']
    compiled_upstream = upstream.check(Path(common['uri']).parent)
    _require(compiled_upstream == read_artifact(common)
        and saved['schema_version'] == 1 and saved['full_original_effects_closed'] is False
        and benchmark['plans'] == saved['plans'] and benchmark['packages'] == saved['packages']
        and selection == read_artifact(saved['decision']) and selection['benchmark'] == saved['benchmark']
        and selection['recommended_arm'] == saved['selected']
        and selection['selected_width'] == saved['selected_native_width'],
        'published profile compilation or performance choice differs from cold replay')
    if 'width_model' in saved:
        _check_width_model(saved['width_model'], saved['plans'], compiled_upstream['canonical_csr'])
    for width, package in saved['packages'].items():
        native = delivery.check(Path(package['uri']).parent)
        profile = read_artifact(saved['plans'][width])
        _require(native == read_artifact(package) and native['profile_plan'] == saved['plans'][width]
            and native['source_seed'] == profile['source_seed'] == saved['source_seed']
            and profile['original_observation'] == saved['original_observation']
            and profile['oracle'] == saved['oracle']
            and {**native['upstream'], 'uri': common['uri']} == common,
            'candidate differs from its published count, upstream or source')
    return saved


def check_entry(path):
    """Replay the admission and check the observed and compiled sides of the entry."""
    from scar.compiler.triangle_profile_family import resolve

    saved = read_artifact(artifact_reference(Path(path).resolve(strict=True), AUTOMATIC))
    read_artifact(saved['family'])
    selected = resolve(saved['family']['uri'], saved['entry']['uri'], interpreter=sys.executable)
    read_artifact(saved['entry'], json_document=False)
    hardware = read_artifact(saved['hardware'])
    acquired = saved['acquisition']
    _check_acquisition(acquired, saved['entry'], saved['argv'])
    compiled = check(saved['compilation']['uri'])
    anchor = Path(saved['anchor']['uri']).parent
    plans = [read_artifact(plan) for plan in compiled['plans'].values()]
    _require(saved['schema_version'] == 1
        and selected['anchor'] == saved['anchor'] and selected['entry'] == saved['entry']
        and selected['source_seed'] == acquired['family']
        and compiled == read_artifact(saved['compilation'])
        and compiled['source_seed'] == acquired['source_seed']
        and compiled['original_observation'] == acquired['original_observation']
        and compiled['oracle'] == acquired['oracle']
        and sorted(map(int, compiled['plans'])) == saved['candidate_widths']
        and hardware['schema'] == profiles.HARDWARE
        and all(plan['hardware'] == saved['hardware']
                and Path(plan['anchor']['package']['uri']).parent == anchor/'digest' for plan in plans),
        'entry admission, observed run and compiled profile differ')
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('anchor', 'source-seed', 'hardware', 'oracle', 'original-observation', 'out'):
        parser.add_argument('--'+flag, type=Path)
    parser.add_argument('--check', type=Path)
    parser.add_argument('--family', type=Path)
    parser.add_argument('--entry', type=Path)
    parser.add_argument('--argv', nargs='+')
    parser.add_argument('--max-capture-seconds', type=float, default=60)
    parser.add_argument('--resume', action='store_true', help='resume offline source projection after a completed capture')
    parser.add_argument('--width', type=int, action='append',
                        help='measure an explicit worker width instead of automatic cost nomination')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args(argv)
    explicit = (args.anchor, args.source_seed, args.hardware, args.oracle, args.original_observation)
    automatic = (args.family, args.entry, args.argv)
    if args.check is not None and not any((*explicit, *automatic, args.out, args.width, args.resume)):
        schema = json.loads(args.check.read_bytes())['schema']
        value = check_entry(args.check) if schema == AUTOMATIC else check(args.check)
    elif args.check is None and args.out and all(automatic) and not any(explicit):
        value = compile_entry(*automatic, args.out, widths=args.width, repeats=args.repeats,
                              max_capture_seconds=args.max_capture_seconds, resume=args.resume)
    elif args.check is None and args.out and all(explicit) and not any(automatic) and args.width is not None and not args.resume:
        value = compile_profile(*explicit, args.out, widths=args.width, repeats=args.repeats)
    else:
        parser.error('use --family/--entry/--argv and --out, the explicit evidence inputs, or --check RESULT')
    print(json.dumps(value, sort_keys=True))


if __name__ == '__main__':
    main()

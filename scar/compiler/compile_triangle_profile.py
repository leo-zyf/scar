"""Plan, lower and validate an admitted fixed-count source triangle profile."""

import argparse
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
from shutil import copytree
from statistics import median
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from scar.codegen import native_parameterized_upstream as upstream
from scar.codegen import native_publication_delivery
from scar.codegen import native_source_triangle_profile as delivery
from scar.codegen.native_ordered_profile import export_profile_solver
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import native_ordered_profile as profiles
from scar.optimization.native_ordered_publication import ENV

SCHEMA = 'scar.source-triangle-profile-compilation'
BENCHMARK = 'scar.source-triangle-profile-benchmark'


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


def _stage_measurement(root, packages):
    """Hold executable and library identity constant wherever arms agree."""
    staged = {}
    for width, reference in packages.items():
        stage = root / width
        copytree(Path(reference['uri']).parent, stage, copy_function=os.link)
        staged[width] = stage
    first, *others = staged.values()
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
    environment = {**ENV, **{key:value for key,value in context['environment'].items() if value is not None}}
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


def compile_profile(anchor, source_seed, hardware, oracle, observation, out, *, widths, repeats=3):
    """Reuse the checked upstream provider family and measure fresh scoped schedules."""
    anchor = Path(anchor).resolve(strict=True)
    baseline = native_publication_delivery.check(anchor)
    _require(baseline['schema_version'] == 6 and type(repeats) is int and repeats >= 3
        and len(widths) >= 2 and len(widths) == len(set(widths)),
        'supply a source-sealed provider family and at least two distinct measured widths')
    out = Path(out).resolve()
    _require(not out.is_relative_to(anchor), 'profile output must be outside the provider family')
    out.mkdir(parents=True, exist_ok=False)
    plans = {}
    for width in sorted(widths):
        schedule = profiles.plan_profile(anchor/'digest', source_seed, hardware, oracle, width,
            original_observation_path=observation)
        plans[str(width)] = _write(out/f'width-{width}.json', schedule)
    documents = [read_artifact(reference) for reference in plans.values()]
    recipe_ids = {document['resources']['recipe_id'] for document in documents}
    _require(len(recipe_ids) == 1, 'schedule widths must share one numerical wire identity')
    common = upstream.export(anchor, source_seed, count=documents[0]['resources']['case']['call_count'],
        recipe_id=recipe_ids.pop(), out=out/'upstream')
    packages = {}
    for width, reference in plans.items():
        solver = out/f'solver-{width}'
        export_profile_solver(reference['uri'], solver)
        packages[width] = delivery.export(reference['uri'], out/'upstream', solver, out/f'native-{width}')
    benchmark = _write(out/'benchmark.json', _measure(out, plans, packages, repeats=repeats))
    selection = profiles.select_profile(benchmark['uri'])
    decision = _write(out/'selection.json', selection)
    result = dict(schema=SCHEMA, schema_version=1, source_seed=documents[0]['source_seed'],
        original_observation=documents[0]['original_observation'], oracle=documents[0]['oracle'],
        upstream=common, plans=plans, packages=packages, benchmark=benchmark, decision=decision,
        selected=selection['recommended_arm'], selected_native_width=selection['selected_width'],
        full_original_effects_closed=False)
    return _write(out/'result.json', result)


def check(path):
    """Replay the prepared artifacts and selection without timing the workload again."""
    reference = artifact_reference(Path(path).resolve(strict=True), SCHEMA)
    saved = read_artifact(reference)
    benchmark = read_artifact(saved['benchmark'])
    selection = profiles.verify_profile_selection(saved['decision']['uri'])
    common = saved['upstream']
    _require(upstream.check(Path(common['uri']).parent) == read_artifact(common)
        and saved['schema_version'] == 1 and saved['full_original_effects_closed'] is False
        and benchmark['plans'] == saved['plans'] and benchmark['packages'] == saved['packages']
        and selection == read_artifact(saved['decision']) and selection['benchmark'] == saved['benchmark']
        and selection['recommended_arm'] == saved['selected']
        and selection['selected_width'] == saved['selected_native_width'],
        'published profile compilation or performance choice differs from cold replay')
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('anchor', 'source-seed', 'hardware', 'oracle', 'original-observation', 'out'):
        parser.add_argument('--'+flag, type=Path)
    parser.add_argument('--check', type=Path)
    parser.add_argument('--width', type=int, action='append')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args(argv)
    sources = (args.anchor, args.source_seed, args.hardware, args.oracle, args.original_observation, args.out)
    if args.check is not None and not any(sources) and args.width is None:
        value = check(args.check)
    elif args.check is None and all(source is not None for source in sources) and args.width is not None:
        value = compile_profile(*sources, widths=args.width, repeats=args.repeats)
    else:
        parser.error('use --check RESULT or provide the admitted sources, widths and output directory')
    print(json.dumps(value, sort_keys=True))


if __name__ == '__main__':
    main()

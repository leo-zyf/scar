"""Measure isolated lane service using the proved grouped CPU numerical body."""

import argparse
import json
from os import sched_getaffinity
from pathlib import Path
import subprocess

from scar.analysis.cpu_array_native_route import SCHEMA as PROOF, _machine_identity, verify_cpu_array_native_route
from scar.analysis.replay_transaction import replay_transaction
from scar.codegen.native_cpu_arena_reuse import _numerical_links
from scar.optimization import independent_array_task_schedule as base
from scar.optimization import independent_array_task_team_schedule as team
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.cpu-array-native-route-body-observation'
_ROUTE = 'grouped_openmp'
_SOURCES = {'probe.cpp': team._PROBE.encode,
            'ordered_openmp.hpp': team._OPENMP_HEADER.read_bytes,
            'ordered_processes.hpp': team._PROCESS_HEADER.read_bytes}


def _need(condition, reason):
    if not condition:
        raise ValueError('CPU array lane calibration: ' + reason)


def _source(proof_ref):
    return _source_from_proof(verify_cpu_array_native_route(proof_ref))


def _source_from_proof(proof):
    schedule = read_artifact(proof['evidence']['schedule'])
    family = read_artifact(schedule['process_plan'])
    runtime_ref = proof['grouped_runtime']
    runtime = read_artifact(runtime_ref)
    grouped_ref = proof['baseline_grouped']
    grouped = read_artifact(grouped_ref)
    environment = proof['workload']['openmp_grouped_environment']
    recipe = read_artifact(grouped['recipe'])
    _need(grouped['runtime'] == runtime_ref and grouped['decision'] == runtime['decision']
          and runtime['decision'] in schedule['alternatives'] and runtime['decision']['mode'] == team.MODE
          and grouped['compiler']['uri'] == str(Path('/usr/bin/g++').resolve(strict=True))
          and proof['scope'] == schedule['scope'] == family['scope'] == grouped['scope']
          and team._environment(recipe) == environment,
          'exact grouped dispatch, C++ toolchain and environment are required')
    for local, name in (('ordered_openmp.hpp', 'native_ordered_openmp'),
                        ('ordered_processes.hpp', 'native_ordered_dispatch')):
        header = grouped['files'][f'include/scar_generated/{name}.hpp']
        _need(read_artifact(header, json_document=False) == _SOURCES[local](),
              'isolated dispatch differs from the qualified release header')
    return proof, family, grouped, runtime, environment


def _seeds(seeds, proof, family, grouped):
    serial = read_artifact(family['native'])
    source = dict(grouped, seed_option=serial['seed_option'], seed_range=serial['seed_range'])
    team._seeds(seeds, family, source)
    schedule = read_artifact(proof['evidence']['schedule'])
    earlier = read_artifact(schedule['thread_evidence'])['seeds']
    holdout = read_artifact(proof['workload']['holdout'])
    width = family['scope']['task_count']
    seed_key = serial['seed_option']
    _need(all(abs(seed - other) >= width for seed in seeds for other in (*earlier, holdout['parameters'][seed_key])),
          'calibration seeds must be disjoint from prior team probes and the validation case')
    return source


def _rows(output, runtime, family, seeds):
    choice = runtime['decision']
    return [(seed, part, team._command(str(output/'probe'), choice, family['scope'], seed, part))
            for seed in seeds for part in ('dispatch', 'kernel')]


def _report(proof_ref, source, output, seeds, samples):
    proof, family, grouped, runtime, environment = source
    grouped_body = _seeds(seeds, proof, family, grouped)
    _need([(row['seed'], row['part'], row['command']) for row in samples]
          == _rows(output, runtime, family, seeds), 'calibration command sequence changed')
    compiler = team._compile_command(output, grouped)
    binary = str(output/'probe')
    compiler_ref, openmp = team._runtime_identity(binary, compiler[0], grouped, environment)
    _need(compiler_ref == grouped['compiler']
          and openmp == grouped['linkage']['qualified_runtime_dependencies']['libgomp.so.1'],
          'calibration must link the exact grouped OpenMP runtime and compiler')
    numerical = _numerical_links(runtime, binary)
    release_numerical = grouped['linkage']['numerical']
    _need(_machine_identity(numerical) == _machine_identity(release_numerical),
          'isolated service probe altered linked numerical provider code')
    evidence = dict(schema_version=1, boundary=team._SAMPLE_BOUNDARY, thread_environment=environment,
                    compiler=compiler, compiler_binary=compiler_ref, openmp_runtime=openmp, seeds=seeds,
                    probe=base._ref(binary, 'application/octet-stream'),
                    sources={name:base._ref(output/name, 'text/x-c++') for name in _SOURCES}, samples=samples)
    key = team.MODE, runtime['decision']['worker_count']
    # Replay the original probe format with the grouped source recorded by this
    # receipt. Shape, seed option and tape originate in the retained serial plan.
    intervals = team._intervals(evidence, family, grouped_body, [runtime['decision']])
    _need(set(intervals) == {key}, 'isolated calibration did not cover its selected lane set')
    return dict(schema=SCHEMA, schema_version=1, proof=proof_ref, route=_ROUTE, executable=grouped['executable'],
                cpu_ids=grouped['cpu_ids'], worker_cpu_ids=runtime['decision']['worker_cpu_ids'],
                scope=proof['scope'], grouped=proof['baseline_grouped'], evidence=evidence,
                numerical_body=grouped['files']['src/profile_0.cpp'], tape=grouped['files']['src/tape_0.inc'],
                conditional_model=intervals[key], task_preparation_and_allocation_in_service=True,
                empty_dispatch_is_proxy=True, whole_entry_executed=False, deployment_authorized=False)


def measure(proof_ref, output, *, seeds):
    source = _source(proof_ref)
    proof, family, grouped, runtime, environment = source
    _seeds(seeds, proof, family, grouped)
    _need(set(grouped['cpu_ids']) <= sched_getaffinity(0), 'recorded CPU affinity is unavailable')
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name, contents in _SOURCES.items():
        (out/name).write_bytes(contents())
    subprocess.run(team._compile_command(out, grouped), cwd=out, env=environment, check=True, timeout=120)
    mode, width = team.MODE, runtime['decision']['worker_count']
    samples = []
    for index, (seed, part, command) in enumerate(_rows(out, runtime, family, seeds)):
        stdout = base._capture(command, env=environment, stdout=out/f'{index:02d}-{part}.txt')
        samples.append(dict(seed=seed, part=part, command=command, width=width, mode=mode, stdout=stdout))
    return base._write(out/'result.json', _report(proof_ref, source, out, seeds, samples))


def verify_cpu_array_native_route_body(reference):
    return _verify(reference)


def _verify(reference, proof_ref=None, proof=None):
    report = read_artifact(reference)
    _need(reference['schema'] == SCHEMA and report['schema_version'] == 1, 'unsupported lane calibration')
    if proof is None:
        proof_ref = report['proof']
        source = _source(proof_ref)
    else:
        _need(proof_ref == report['proof'], 'grouped lane calibration belongs to a different proof')
        source = _source_from_proof(proof)
    evidence = report['evidence']
    out = Path(reference['uri']).parent
    _need(report == _report(proof_ref, source, out, evidence['seeds'], evidence['samples']),
          'grouped lane calibration differs at replay')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--proof', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--seed', action='append', type=int, default=[])
    args = parser.parse_args(argv)
    with replay_transaction():
        if args.check is not None:
            if args.out is not None or args.seed:
                parser.error('--check reads an existing lane calibration')
            report = verify_cpu_array_native_route_body(base._ref(args.check, SCHEMA))
        else:
            if args.out is None or len(args.seed) < 4:
                parser.error('--proof requires --out and at least four independent seeds')
            report = read_artifact(measure(base._ref(args.proof, PROOF), args.out, seeds=args.seed))
    print(json.dumps(report['conditional_model']))


if __name__ == '__main__':
    raise SystemExit(main())

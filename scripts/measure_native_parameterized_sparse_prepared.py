"""Measure whole-process startup and solver time with or without invariant preparation."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from measure_native_parameterized_sparse_entry import _trial
from measure_native_parameterized_sparse_width import _arm

from scar.codegen.native_parameterized_sparse_prepared import SCHEMA
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.native_parameterized_sparse_width import _context


BENCHMARK = 'scar.native-parameterized-sparse-prepared-benchmark'
_ARMS = ('original', 'native', 'prepared')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--counts', nargs='+', type=int, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()

    manifest = artifact_reference(args.prepared.resolve(strict=True), SCHEMA)
    prepared = read_artifact(manifest)
    package = Path(manifest['uri']).parent
    optimized = dict(prepared['binary'], uri=str(package / prepared['binary']['uri']))
    read_artifact(optimized, json_document=False)
    native, plan = _arm(Path(prepared['entry']['uri']))
    scope = plan['scope']
    cpus = plan['resources']['placement']['launch_cpu_ids']
    if (native['artifact'] != prepared['entry'] or scope != prepared['scope']
            or len(set(args.counts)) != len(args.counts) or args.rounds < 3
            or not set(cpus) <= os.sched_getaffinity(0)
            or any(not scope['minimum_count'] <= n <= scope['maximum_count'] for n in args.counts)):
        parser.error('expected matched artifacts, admitted counts, all named CPUs and at least three clean rounds')
    source = scope['source']
    read_artifact(source, json_document=False)
    python, cwd, environment = _context(plan)
    version = subprocess.check_output([python, '-B', '-c', 'import sys; print(sys.version)'],
                                      cwd=cwd, env=environment, text=True).strip()
    arms = dict(original=dict(source=source), native=native,
                prepared=dict(artifact=manifest, executable=optimized))
    cohort = ['/usr/bin/taskset', '-c', ','.join(map(str, cpus))]
    output = args.out.resolve()
    output.mkdir()
    cases = []
    for count in args.counts:
        case = output / str(count)
        case.mkdir()
        argv = [str(count)]
        commands = dict(original=[*cohort, python, '-B', source['uri'], *argv],
                        native=[*cohort, native['executable']['uri'], *argv],
                        prepared=[*cohort, optimized['uri'], *argv])
        rows = []
        output_digest = None
        for round_ in range(args.rounds + 1):
            start = round_ % len(_ARMS)
            for arm in (*_ARMS[start:], *_ARMS[:start]):
                row = _trial(arm, commands[arm], round_, cwd, environment, case)
                digest = row['stdout']['sha256']
                if output_digest is not None and digest != output_digest:
                    raise ValueError(f'prepared, full native and original outputs differ at {count}')
                output_digest = digest
                rows.append(row)
        cases.append(dict(count=count, argv=argv, rows=rows))
    report = dict(schema=BENCHMARK, schema_version=1, source=source, cpu_ids=cpus, cwd=cwd,
                  environment=environment, interpreter=dict(python_executable=python, python_version=version),
                  arms=arms, cases=cases, launcher=artifact_reference(Path(__file__).resolve(), 'scar.python-source'))
    pending = output / '.run.json.tmp'
    with pending.open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    pending.rename(output / 'run.json')


if __name__ == '__main__':
    main()

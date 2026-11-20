"""Measure two native sparse widths against the same runtime counts and source."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from measure_native_parameterized_sparse_entry import _trial

from scar.codegen.native_parameterized_sparse_entry import SCHEMA
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.native_parameterized_sparse_width import BENCHMARK, _context


def _arm(path):
    reference = artifact_reference(path.resolve(strict=True), SCHEMA)
    artifact = read_artifact(reference)
    package = Path(reference['uri']).parent
    binary = dict(artifact['binary'], uri=str(package / artifact['binary']['uri']))
    read_artifact(binary, json_document=False)
    return dict(artifact=reference, executable=binary), read_artifact(artifact['parameterized_plan'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--w2', type=Path, required=True)
    parser.add_argument('--w6', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--counts', type=int, nargs='+', required=True)
    parser.add_argument('--pairs', type=int, default=3)
    args = parser.parse_args()

    w2, narrow = _arm(args.w2)
    w6, wide = _arm(args.w6)
    arms = {'2': w2, '6': w6}
    if [narrow['resources']['width'], wide['resources']['width']] != [2, 6]:
        parser.error('expected W2 and W6 package arguments in that order')
    scope = wide['scope']
    cpus = wide['resources']['placement']['launch_cpu_ids']
    if (narrow['scope'] != scope or args.pairs < 3 or len(set(args.counts)) != len(args.counts)
            or not set(cpus) <= os.sched_getaffinity(0)
            or any(not scope['minimum_count'] <= count <= scope['maximum_count'] for count in args.counts)):
        parser.error('expected admitted counts, all named CPUs and at least three pairs')
    python, cwd, environment = _context(wide)
    if _context(narrow) != (python, cwd, environment):
        parser.error('both arms must bind the same launch contract')
    source = scope['source']
    read_artifact(source, json_document=False)
    version = subprocess.check_output([python, '-B', '-c', 'import sys; print(sys.version)'],
                                      cwd=cwd, env=environment, text=True).strip()
    cohort = ['/usr/bin/taskset', '-c', ','.join(map(str, cpus))]
    output = args.out.resolve()
    output.mkdir()
    cases = []
    for count in args.counts:
        case = output / str(count)
        case.mkdir()
        argv = [str(count)]
        original = _trial('original', [*cohort, python, '-B', source['uri'], *argv], 0, cwd, environment, case)
        rows = []
        for round_ in range(args.pairs + 1):
            for width in ((2, 6) if round_ % 2 == 0 else (6, 2)):
                executable = arms[str(width)]['executable']['uri']
                row = _trial(f'w{width}', [*cohort, executable, *argv], round_, cwd, environment, case)
                if row['stdout']['sha256'] != original['stdout']['sha256']:
                    raise ValueError(f'W{width} and original output differ for count {count}')
                row.pop('arm')
                rows.append(dict(row, width=width))
        cases.append(dict(count=count, argv=argv, original=original, rows=rows))
    report = dict(schema=BENCHMARK, schema_version=1, source=source, cpu_ids=cpus, cwd=cwd,
                  environment=environment, interpreter=dict(python_executable=python, python_version=version),
                  arms=arms, cases=cases, launcher=artifact_reference(Path(__file__).resolve(), 'scar.python-source'))
    with (output / 'run.json').open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()

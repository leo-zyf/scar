"""Compare an original sparse application and its native runtime-count edition."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from time import perf_counter_ns

from scar.codegen import native_parameterized_sparse_entry as native
from scar.compiler.compile_triangle_profile import _launch_environment
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.native-parameterized-sparse-paired-run'
_ARMS = ('original','entry_native')
_DIGEST = re.compile(rb'[0-9a-f]{64}\n')


def _trial(role, argv, round_, cwd, environment, out):
    started = perf_counter_ns()
    child = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, env=environment, cwd=cwd, timeout=125)
    elapsed = perf_counter_ns() - started
    stdout, stderr = (out / f'{round_:02d}-{role}.{extension}' for extension in ('stdout','stderr'))
    stdout.write_bytes(child.stdout)
    stderr.write_bytes(child.stderr)
    if child.returncode or child.stderr or not _DIGEST.fullmatch(child.stdout):
        raise ValueError(f'invalid output or status from {role}; see {stderr}')
    print(f'{out.name} {role} {round_} {elapsed / 1e9:.3f}s', flush=True)
    return dict(arm=role, round=round_, kind='OBSERVED_WARMUP' if round_ == 0 else 'CLEAN_MEASURED',
                wall_ns=elapsed, returncode=child.returncode, command=argv,
                stdout=artifact_reference(stdout,'application/octet-stream'),
                stderr=artifact_reference(stderr,'application/octet-stream'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--counts',nargs='+',type=int,required=True)
    parser.add_argument('--pairs',type=int,default=3)
    args = parser.parse_args()
    ref = artifact_reference(args.artifact.resolve(strict=True),native.SCHEMA)
    artifact = read_artifact(ref)
    plan_ref = artifact['parameterized_plan']
    plan = read_artifact(plan_ref)
    cpus = plan['resources']['placement']['launch_cpu_ids']
    scope = plan['scope']
    if (args.pairs < 3 or len(set(args.counts)) != len(args.counts) or not set(cpus) <= os.sched_getaffinity(0)
            or any(not scope['minimum_count'] <= size <= scope['maximum_count'] for size in args.counts)):
        parser.error('expected unique admitted counts, three paired samples and all named CPUs')
    source = scope['source']
    read_artifact(source,json_document=False)
    profile = read_artifact(plan['artifacts']['fixed_profile_plan'])
    context = read_artifact(profile['anchor']['release'])['context']
    cwd = context['cwd']
    environment = _launch_environment(context)
    interpreter = context['interpreter']['executable']
    version = subprocess.check_output([interpreter,'-B','-c','import sys; print(sys.version)'],
                                     env=environment,cwd=cwd,text=True).strip()
    binary = Path(ref['uri']).parent / artifact['binary']['uri']
    binary = dict(artifact['binary'],uri=str(binary.resolve(strict=True)))
    read_artifact(binary,json_document=False)
    arms = dict(original=dict(python_executable=str(Path(interpreter).resolve(strict=True)),python_version=version),
                entry_native=dict(artifact=ref,executable=binary))
    cohort = ['/usr/bin/taskset','-c',','.join(map(str, cpus))]
    output = args.out.resolve()
    output.mkdir()
    cases = []
    for count in args.counts:
        case = output / str(count)
        case.mkdir()
        argv = [str(count)]
        commands = dict(original=[*cohort,interpreter,'-B',source['uri'],*argv],
                        entry_native=[*cohort,binary['uri'],*argv])
        rows = []
        expected = None
        for round_ in range(args.pairs + 1):
            for arm in _ARMS if round_%2 == 0 else reversed(_ARMS):
                row = _trial(arm,commands[arm],round_,cwd,environment,case)
                match = row['stdout']['sha256']
                if expected is not None and expected != match:
                    raise ValueError(f'fresh original and native output differ for count {count}')
                expected = match
                rows.append(row)
        cases.append(dict(count=count,argv=argv,rows=rows))
    report = dict(schema=SCHEMA,schema_version=1,parameterized_plan=plan_ref,source=source,cpu_ids=cpus,
                  cwd=cwd,environment=environment,arms=arms,cases=cases,
                  launcher=artifact_reference(Path(__file__).resolve(),'scar.python-source'))
    with (output/'run.json').open('x',encoding='utf8') as stream:
        json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()

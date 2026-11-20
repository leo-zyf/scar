"""Compare the parameterized native entry with a direct original process."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns

from scar.analysis.scalar_recursive_driver import CAPTURE
from scar.codegen import scalar_recursive_parameterized as native
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.scalar_sorted_transition_cost import NORMALIZATION, canonical_stdout


SCHEMA = 'scar.scalar-recursive-parameterized-paired-run'
ARMS = ('original', 'entry_native')


def launch(arm, setups, capture, source, protocol, environment, out, cpu, ordinal):
    prefix = out / f'{ordinal:02d}-{arm}'
    stdout, stderr = (prefix.with_suffix(suffix) for suffix in ('.stdout','.stderr'))
    command = ['/usr/bin/taskset','-c',str(cpu),'/usr/bin/timeout','-k','2','60']
    observations = None
    if arm == 'original':
        command.extend((sys.executable, '-B', '-u', str(source)))
    else:
        command.append(setups[arm]['executable']['uri'])
        if ordinal == 0:
            observations = out / 'native-observation'
            observations.mkdir()
            command.extend(('--observe', str(observations)))
    start = perf_counter_ns()
    with stdout.open('xb') as output, stderr.open('xb') as errors:
        with subprocess.Popen(command, cwd=source.parent, env=environment, stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=errors) as process:
            for text, marker in protocol:
                process.stdin.write(text)
                process.stdin.flush()
                while marker is not None:
                    line = process.stdout.readline()
                    if not line:
                        raise ValueError(f'{arm} exited before {marker!r}; see {stderr}')
                    output.write(line)
                    if line.startswith(marker):
                        break
            rest, _ = process.communicate(timeout=9)
            output.write(rest)
    elapsed = perf_counter_ns() - start
    if process.returncode or stderr.stat().st_size:
        raise ValueError(f'{arm} exited with status {process.returncode}; see {stderr}')
    observation = None
    if observations is not None:
        result = native.project_observation(setups[arm]['artifact']['uri'], observations, capture['uri'])
        path = observations / 'observation.json'
        with path.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        observation = artifact_reference(path, native.OBSERVATION)
    print(f'{arm} {ordinal} {elapsed / 1e9:.3f}s', flush=True)
    return dict(arm=arm, round=ordinal, kind='OBSERVED_WARMUP' if ordinal == 0 else 'CLEAN_MEASURED',
                wall_ns=elapsed, returncode=process.returncode, command=command, observation=observation,
                stdout=artifact_reference(stdout,'application/octet-stream'),
                stderr=artifact_reference(stderr,'application/octet-stream'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifact','original-capture','out'):
        parser.add_argument('--' + name,type=Path,required=True)
    parser.add_argument('--cpu',type=int,required=True)
    parser.add_argument('--pairs',type=int,default=4)
    args = parser.parse_args()
    if args.pairs < 3 or args.cpu not in os.sched_getaffinity(0):
        parser.error('three measured pairs and an available logical CPU are required')
    native_ref = artifact_reference(args.artifact.resolve(strict=True),native.SCHEMA)
    artifact = read_artifact(native_ref)
    plan_ref = artifact['parameterized_plan']
    recipe = read_artifact(plan_ref)
    source_ref = recipe['scope']['source']
    source = Path(source_ref['uri'])
    read_artifact(source_ref,json_document=False)
    capture = artifact_reference(args.original_capture.resolve(strict=True),CAPTURE)
    captured = read_artifact(capture)
    if captured['source'] != source_ref or captured['frontiers']:
        raise ValueError('held-out source observation is from a different or incomplete program')
    expected_stdout = canonical_stdout(read_artifact(captured['stdout'],json_document=False))
    captured_tape = read_artifact(captured['stdin'],json_document=False)
    if not captured_tape.endswith(b'\n'):
        raise ValueError('expected a newline-terminated held-out command stream')
    protocol = []
    operations = {row['kind']:row['tokens'] for row in recipe['grammar']['commands']}
    replies = {tuple(operations[kind]):marker for kind, marker in (('UCI',b'uciok'),('ISREADY',b'readyok'))}
    for raw in captured_tape.splitlines(keepends=True):
        tokens = raw.decode('ascii').split()
        marker = b'bestmove ' if tokens[:2] == operations['GO_DEPTH'] else replies.get(tuple(tokens))
        protocol.append((raw,marker))
    exe = (Path(native_ref['uri']).parent / artifact['binary']['uri']).resolve(strict=True)
    binary = dict(artifact['binary'],uri=str(exe))
    read_artifact(binary,json_document=False)
    setups = dict(original=dict(python_executable=str(Path(sys.executable).resolve()),python_version=sys.version),
                  entry_native=dict(artifact=native_ref,executable=binary))
    environment = {key:os.environ[key] for key in ('PATH','PYTHONPATH') if key in os.environ}
    environment.update(LANG='C.UTF-8',LC_ALL='C',PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1')
    out = args.out.resolve()
    out.mkdir()
    rows = []
    for ordinal in range(args.pairs + 1):
        for arm in ARMS if ordinal % 2 == 0 else reversed(ARMS):
            result = launch(arm,setups,capture,source,protocol,environment,out,args.cpu,ordinal)
            if canonical_stdout(read_artifact(result['stdout'],json_document=False)) != expected_stdout:
                raise ValueError(f'{arm} round {ordinal} differs from the independently observed original')
            rows.append(result)
    report = dict(schema=SCHEMA,schema_version=1,parameterized_plan=plan_ref,source=source_ref,
                  stdin=captured['stdin'],original_capture=capture,normalization=NORMALIZATION,
                  cpu_ids=[args.cpu],cwd=str(source.parent),argv=[],environment=environment,arms=setups,rows=rows,
                  launcher=artifact_reference(Path(__file__).resolve(),'scar.python-source'))
    with (out / 'run.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()

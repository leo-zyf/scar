"""Compare direct original and standalone C++20 entry processes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns

from scar.codegen import scalar_recursive_entry as entry
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.scalar_sorted_transition_cost import NORMALIZATION, canonical_stdout


SCHEMA = 'scar.scalar-recursive-entry-paired-run'
ARMS = ('original', 'entry_native')


def launch(arm, setup, publication, round_, out, cpu, environment, source):
    prefix = out / f'{round_:02d}-{arm}'
    stdout, stderr = (prefix.with_suffix(suffix) for suffix in ('.stdout', '.stderr'))
    command = ['/usr/bin/taskset', '-c', str(cpu), '/usr/bin/timeout', '-k', '2', '60']
    observations = None
    if arm == 'original':
        command.extend((sys.executable, '-B', '-u', str(source)))
    else:
        command.append(setup['executable']['uri'])
        if round_ == 0:
            observations = out / 'native-observation'
            observations.mkdir()
            command.extend(('--observe', str(observations)))
    started = perf_counter_ns()
    with stdout.open('xb') as output, stderr.open('xb') as errors:
        with subprocess.Popen(command, cwd=source.parent, env=environment, stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=errors) as child:
            for text, marker in publication:
                child.stdin.write(text)
                child.stdin.flush()
                while marker:
                    line = child.stdout.readline()
                    if not line:
                        raise ValueError(f'child exited before {marker!r}; see {stderr}')
                    output.write(line)
                    if line.startswith(marker):
                        break
            remaining, _ = child.communicate(timeout=9)
            output.write(remaining)
    elapsed = perf_counter_ns() - started
    if child.returncode or stderr.stat().st_size:
        raise ValueError(f'direct source process failed; see {stderr}')
    observation = None
    if observations is not None:
        manifest = entry.project_observation(setup['artifact']['uri'], observations)
        path = observations / 'observation.json'
        with path.open('x', encoding='utf8') as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        observation = artifact_reference(path, entry.OBSERVATION)
    print(f'{arm} {round_} {elapsed / 1e9:.3f}s', flush=True)
    return dict(arm=arm, round=round_, kind='OBSERVED_WARMUP' if round_ == 0 else 'CLEAN_MEASURED',
                wall_ns=elapsed, returncode=child.returncode, command=command, observation=observation,
                stdout=artifact_reference(stdout, 'application/octet-stream'),
                stderr=artifact_reference(stderr, 'application/octet-stream'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--pairs', type=int, default=4)
    args = parser.parse_args()
    if args.pairs < 2 or args.cpu not in os.sched_getaffinity(0):
        parser.error('two measured pairs and an available logical CPU are required')
    art_ref = artifact_reference(args.artifact.resolve(strict=True), entry.SCHEMA)
    native = read_artifact(art_ref)
    plan_ref = native['entry_plan']
    plan = read_artifact(plan_ref)
    scope = plan['scope']
    source = Path(scope['source']['uri'])
    read_artifact(scope['source'], json_document=False)
    binary = (Path(art_ref['uri']).parent / native['binary']['uri']).resolve(strict=True)
    binary_ref = dict(native['binary'], uri=str(binary))
    read_artifact(binary_ref, json_document=False)
    publication = []
    for row in plan['commands']:
        marker = None
        if row['operation'] == 'STATIC_REPLY':
            marker = (row['reply_lines'][-1] + '\n').encode('ascii')
        elif row['operation'] == 'SEARCH':
            marker = b'bestmove '
        publication.append(((row['line'] + '\n').encode('ascii'), marker))
    if b''.join(line for line, _ in publication) != read_artifact(scope['stdin'], json_document=False):
        raise ValueError('paced launcher must use the exact approved source input')
    arms = dict(
        original=dict(python_executable=str(Path(sys.executable).resolve()), python_version=sys.version),
        entry_native=dict(artifact=art_ref, executable=binary_ref))
    env = {key:os.environ[key] for key in ('PATH','PYTHONPATH') if key in os.environ}
    env.update(LANG='C.UTF-8', LC_ALL='C', PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1')
    out = args.out.resolve()
    out.mkdir()
    rows = []
    expected = None
    for round_ in range(args.pairs + 1):
        for arm in ARMS if round_ % 2 == 0 else reversed(ARMS):
            row = launch(arm, arms[arm], publication, round_, out, args.cpu, env, source)
            normalized = canonical_stdout(read_artifact(row['stdout'], json_document=False))
            if expected is not None and normalized != expected:
                raise ValueError(f'canonical CLI output differs on {arm} round {round_}')
            expected = normalized
            rows.append(row)
    receipt = dict(schema=SCHEMA, schema_version=1, entry_plan=plan_ref, source=scope['source'],
                   stdin=scope['stdin'], normalization=NORMALIZATION, cpu_ids=[args.cpu], cwd=str(source.parent),
                   argv=[], environment=env, arms=arms, rows=rows,
                   launcher=artifact_reference(Path(__file__).resolve(), 'scar.python-source'))
    with (out / 'run.json').open('x', encoding='utf8') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()

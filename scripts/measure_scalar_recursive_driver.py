"""Measure paired fixed-depth UCI launches of an admitted scalar driver."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.scalar_recursive_driver_plan import BENCHMARK, SCHEMA as PLAN, STUDY
from scar.optimization.scalar_sorted_transition_cost import NORMALIZATION, canonical_stdout
from scar.runtime import scalar_recursive_driver_study as bridge


ARMS = ('original', 'driver_native')
INPUT_FLAGS = (
    ('library', '--library', '--sha256'),
    ('source_driver', '--source-driver', '--driver-sha256'),
    ('ui_contract', '--ui-contract', '--ui-sha256'),
)


def run_arm(arm, inputs, commands, round_, out, cpu, environment, source):
    prefix = out / f'{round_:02d}-{arm}'
    report = prefix.with_suffix('.json')
    stdout = prefix.with_suffix('.stdout')
    stderr = prefix.with_suffix('.stderr')
    command = ['/usr/bin/taskset', '-c', str(cpu), '/usr/bin/timeout', '-k', '2', '60',
               sys.executable, '-B', '-u', '-m', bridge.__name__, '--report', str(report)]
    if arm == 'driver_native':
        command.append('--isolated-study')
        for key, path_flag, sha_flag in INPUT_FLAGS:
            command.extend((path_flag, inputs[key]['uri'], sha_flag, inputs[key]['sha256']))
        if round_ == 0:
            command.append('--observe')
    command.extend(('--', str(source)))

    started = perf_counter_ns()
    with stdout.open('xb') as output, stderr.open('xb') as errors:
        with subprocess.Popen(command, cwd=source.parent, env=environment, stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=errors) as child:
            for request in commands:
                child.stdin.write((request + '\n').encode('ascii'))
                child.stdin.flush()
                marker = {'uci': b'uciok', 'isready': b'readyok'}.get(request)
                if request.startswith('go '):
                    marker = b'bestmove '
                while marker:
                    line = child.stdout.readline()
                    if not line:
                        raise ValueError(f'engine exited before {marker!r}; see {stderr}')
                    output.write(line)
                    if line.startswith(marker):
                        break
            remaining, _ = child.communicate(timeout=9)
            output.write(remaining)
    elapsed = perf_counter_ns() - started
    if child.returncode or stderr.stat().st_size:
        raise ValueError(f'paired execution failed; see {stderr}')
    print(f'{arm} {round_} {elapsed / 1e9:.3f}s', flush=True)
    return dict(arm=arm, round=round_, kind='OBSERVED_WARMUP' if round_ == 0 else 'CLEAN_MEASURED',
                wall_ns=elapsed, returncode=child.returncode, command=command,
                stdout=artifact_reference(stdout, 'application/octet-stream'),
                stderr=artifact_reference(stderr, 'application/octet-stream'),
                report=artifact_reference(report, STUDY))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'library', 'ui-contract', 'commands', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--pairs', type=int, default=4)
    args = parser.parse_args()
    if args.pairs < 1 or args.cpu not in os.sched_getaffinity(0):
        parser.error('at least one measured pair and an available logical CPU are required')

    plan_ref = artifact_reference(args.plan.resolve(strict=True), PLAN)
    plan = read_artifact(plan_ref)
    source = Path(plan['source']['uri'])
    read_artifact(plan['source'], json_document=False)
    commands = args.commands.read_text(encoding='ascii').splitlines()
    if not commands or commands[-1] != 'quit' or any(
        line.startswith('go ') and not line.removeprefix('go depth ').isdigit() for line in commands
    ):
        parser.error('expected a fixed-depth UCI transcript ending with quit')
    out = args.out.resolve()
    out.mkdir()
    stdin = out / 'commands.stdin'
    stdin.write_text('\n'.join(commands) + '\n', encoding='ascii')

    arms = {
        'original': dict(library=None, source_driver=None, ui_contract=None),
        'driver_native': dict(
            library=artifact_reference(args.library.resolve(strict=True), bridge.LIBRARY),
            source_driver=plan['artifacts']['driver'],
            ui_contract=artifact_reference(args.ui_contract.resolve(strict=True), bridge.UI)),
    }
    implementation = dict(
        bridge=artifact_reference(Path(bridge.__file__).resolve(), 'scar.python-source'),
        launcher=artifact_reference(Path(__file__).resolve(), 'scar.python-source'),
        python_executable=str(Path(sys.executable).resolve()),
        python_version=sys.version)
    environment = {key: os.environ[key] for key in ('PATH', 'PYTHONPATH') if key in os.environ}
    environment.update(PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1', LC_ALL='C')
    rows = []
    expected_output = None
    for round_ in range(args.pairs + 1):
        for arm in ARMS if round_ % 2 == 0 else reversed(ARMS):
            row = run_arm(arm, arms[arm], commands, round_, out, args.cpu, environment, source)
            output = canonical_stdout(read_artifact(row['stdout'], json_document=False))
            if expected_output is not None and output != expected_output:
                raise ValueError(f'paired execution output changed in {arm} round {round_}')
            expected_output = output
            rows.append(row)
    report = dict(schema=BENCHMARK, schema_version=1, driver_plan=plan_ref, source=plan['source'],
                  stdin=artifact_reference(stdin, 'application/octet-stream'), cpu_ids=[args.cpu],
                  argv=[], cwd=str(source.parent), environment=environment, arms=arms, rows=rows,
                  normalization=NORMALIZATION, implementation=implementation)
    with (out / 'run.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()

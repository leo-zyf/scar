"""Compare per-search materialization with a continuous native owner."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns

import measure_scalar_recursive_driver as driver_study
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime import scalar_recursive_owner_study as owner_study


SCHEMA = 'scar.scalar-recursive-owner-paired-run'


def run_owner(inputs, path, round_, out, cpu, environment, source):
    prefix = out / f'{round_:02d}-owner_native'
    report, stdout, stderr = (prefix.with_suffix(suffix) for suffix in ('.json', '.stdout', '.stderr'))
    command = ['/usr/bin/taskset', '-c', str(cpu), '/usr/bin/timeout', '-k', '2', '60',
               sys.executable, '-B', '-u', '-m', owner_study.__name__, '--report', str(report)]
    for role in ('plan', 'library'):
        command.extend(('--' + role, inputs[role]['uri'], '--' + role + '-sha256', inputs[role]['sha256']))
    if round_ == 0:
        command.append('--observe')

    start = perf_counter_ns()
    with stdout.open('xb') as output, stderr.open('xb') as errors:
        with subprocess.Popen(command, cwd=source.parent, env=environment, stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=errors) as child:
            for line in path.read_bytes().splitlines(keepends=True):
                child.stdin.write(line)
                child.stdin.flush()
                request = line.rstrip(b'\n')
                marker = b'bestmove ' if request.startswith(b'go ') else {
                    b'uci': b'uciok', b'isready': b'readyok'}.get(request)
                while marker:
                    result = child.stdout.readline()
                    if not result:
                        raise ValueError(f'resident process exited before {marker!r}; see {stderr}')
                    output.write(result)
                    if result.startswith(marker):
                        break
            remaining, _ = child.communicate(timeout=9)
            output.write(remaining)
    elapsed = perf_counter_ns() - start
    if child.returncode or stderr.stat().st_size:
        raise ValueError(f'resident owner execution failed; see {stderr}')
    print(f'owner_native {round_} {elapsed / 1e9:.3f}s', flush=True)
    return dict(arm='owner_native', round=round_, kind='OBSERVED_WARMUP' if round_ == 0 else 'CLEAN_MEASURED',
                wall_ns=elapsed, returncode=child.returncode, command=command,
                stdout=artifact_reference(stdout, 'application/octet-stream'),
                stderr=artifact_reference(stderr, 'application/octet-stream'),
                report=artifact_reference(report, owner_study.SCHEMA))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'driver-library', 'owner-library', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--pairs', type=int, default=4)
    args = parser.parse_args()
    if args.pairs < 1 or args.cpu not in os.sched_getaffinity(0):
        parser.error('at least one measured pair and an available logical CPU are required')

    plan_ref = artifact_reference(args.plan.resolve(strict=True), owner_study.PLAN)
    plan = read_artifact(plan_ref)
    if plan['decision']['runtime_experiment_authorized'] is not True:
        parser.error('owner plan must admit the isolated experiment')
    scope = plan['scope']
    source = Path(scope['source']['uri'])
    read_artifact(scope['source'], json_document=False)
    commands = read_artifact(scope['stdin'], json_document=False).decode('ascii').splitlines()
    path = Path(scope['stdin']['uri'])
    ui, source_driver = scope['ui_contract'], scope['source_driver']
    arms = dict(
        driver_native=dict(library=artifact_reference(args.driver_library.resolve(strict=True), driver_study.bridge.LIBRARY),
                           source_driver=source_driver, ui_contract=ui),
        owner_native=dict(library=artifact_reference(args.owner_library.resolve(strict=True), driver_study.bridge.LIBRARY),
                          plan=plan_ref))
    implementation = dict(
        launcher=artifact_reference(Path(__file__).resolve(), 'scar.python-source'),
        driver_bridge=artifact_reference(Path(driver_study.bridge.__file__).resolve(), 'scar.python-source'),
        owner_bridge=artifact_reference(Path(owner_study.__file__).resolve(), 'scar.python-source'),
        python_executable=str(Path(sys.executable).resolve()), python_version=sys.version)
    environment = {key: os.environ[key] for key in ('PATH', 'PYTHONPATH') if key in os.environ}
    environment.update(PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1', LC_ALL='C')
    out = args.out.resolve()
    out.mkdir()
    rows = []
    expected = None
    for round_ in range(args.pairs + 1):
        roles = ('driver_native', 'owner_native')
        for role in roles if round_ % 2 == 0 else reversed(roles):
            if role == 'driver_native':
                row = driver_study.run_arm(role, arms[role], commands, round_, out, args.cpu, environment, source)
            else:
                row = run_owner(arms[role], path, round_, out, args.cpu, environment, source)
            output = driver_study.canonical_stdout(read_artifact(row['stdout'], json_document=False))
            if expected is not None and output != expected:
                raise ValueError(f'paired execution output changed in {role} round {round_}')
            expected = output
            rows.append(row)
    report = dict(schema=SCHEMA, schema_version=1, owner_plan=plan_ref, source=scope['source'],
                  stdin=scope['stdin'], cpu_ids=[args.cpu], argv=[], cwd=str(source.parent),
                  environment=environment, arms=arms, rows=rows,
                  normalization=driver_study.NORMALIZATION, implementation=implementation)
    with (out / 'run.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()

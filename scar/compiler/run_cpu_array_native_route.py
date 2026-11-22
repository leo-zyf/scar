"""Execute one explicitly selected C++20 CPU route from its sealed launch package."""

import argparse
import json
import os
from pathlib import Path
import signal
import stat
import subprocess

from scar.codegen.cpu_array_native_route_package import SCHEMA as PACKAGE, prepare_cpu_array_native_route_launch
from scar.compiler.calibrate_array_entry import _normalize, _write
from scar.compiler.entry_delivery import _environment_digest
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.cpu-array-native-route-execution'


def _capture(path, data, schema):
    with path.open('xb') as stream:
        stream.write(data)
    return artifact_reference(path, schema)


def _publication(path, stdout, oracle, runtime_fields):
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            return None, False, False
    except FileNotFoundError:
        return None, False, False
    reference = artifact_reference(path, 'application/json')
    payload = read_artifact(reference, json_document=False)
    try:
        same_report = json.loads(stdout) == json.loads(payload)
        matched = (_normalize(stdout, {'runtime_fields': runtime_fields}) == oracle
                   and _normalize(payload, {'runtime_fields': runtime_fields}) == oracle)
    except (ValueError, TypeError):
        return reference, False, False
    return reference, same_report, matched


def execute_cpu_array_native_route(reference, *, argv, cpu_ids, target, environment, fresh_root,
                                   prefer_original_native=False):
    """Verify one launch, reserve its fresh cwd, execute once, and retain output evidence."""
    prepared = prepare_cpu_array_native_route_launch(
        reference, argv=argv, cpu_ids=cpu_ids, target=target, environment=environment,
        fresh_root=fresh_root, prefer_original_native=prefer_original_native)
    if prepared['status'] == 'DECLINE_NATIVE_ROUTE':
        return prepared

    invocation = prepared['execution']
    root = Path(invocation['cwd'])
    publication = Path(prepared['publication'])
    oracle = read_artifact(prepared['oracle'], json_document=False)
    root.mkdir(mode=0o700)
    with subprocess.Popen(invocation['argv'], cwd=root, env=invocation['environment'],
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          start_new_session=True) as process:
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=invocation['timeout_seconds'])
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()

    stdout_ref = _capture(root/'stdout.json', stdout, 'application/octet-stream')
    stderr_ref = _capture(root/'stderr.log', stderr, 'application/octet-stream')
    file_ref, same_report, matched = _publication(publication, stdout, oracle, prepared['runtime_fields'])
    validated = not timed_out and process.returncode == 0 and not stderr and same_report and matched
    report = dict(schema=SCHEMA, schema_version=1,
        status='SCOPED_NATIVE_OUTPUT_VALIDATED' if validated else 'SCOPED_NATIVE_VALIDATION_FAILED',
        package=reference, route=prepared['route'], executable=prepared['executable'],
        scope=dict(argv=argv, cpu_ids=cpu_ids, target=target,
                   source_environment_sha256=_environment_digest(environment)),
        launch=dict(command=invocation['argv'], cwd=str(root), exclusive_cwd=True,
                    environment_sha256=_environment_digest(invocation['environment']),
                    timeout_seconds=invocation['timeout_seconds']),
        returncode=process.returncode, timed_out=timed_out, stdout=stdout_ref, stderr=stderr_ref,
        published=file_ref, oracle=prepared['oracle'], runtime_fields=prepared['runtime_fields'],
        stdout_matches_published_report=same_report, normalized_reports_match_oracle=matched,
        stderr_empty=not stderr, target_executed=True, performance_measured=False,
        original_python_action=prepared['original_python_action'], deployment_authorized=False)
    _write(root/'result.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path, help='new run directory under an existing parent')
    parser.add_argument('--incumbent', action='store_true', help='explicitly run the sealed incumbent instead')
    parser.add_argument('entry_argv', nargs=argparse.REMAINDER, help='optional exact entry arguments after --')
    args = parser.parse_args(argv)
    reference = artifact_reference(args.package.resolve(strict=True), PACKAGE)
    package = read_artifact(reference)
    supplied = args.entry_argv if args.entry_argv else package['argv']
    if supplied[:1] == ['--']:
        supplied = supplied[1:]
    result = execute_cpu_array_native_route(
        reference, argv=supplied, cpu_ids=package['cpu_ids'],
        target=package['hardware'], environment=package['source_environment'],
        fresh_root=args.out, prefer_original_native=args.incumbent)
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] == 'SCOPED_NATIVE_OUTPUT_VALIDATED' else 2


if __name__ == '__main__':
    raise SystemExit(main())

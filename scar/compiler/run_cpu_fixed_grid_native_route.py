"""Run one explicitly admitted fixed-grid C++20 route and retain its observables."""

from pathlib import Path
import shutil
import stat
import subprocess

from scar.codegen.cpu_fixed_grid_native_route_package import prepare_cpu_fixed_grid_native_route_launch
from scar.compiler.calibrate_array_entry import _ref, _write
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.cpu-fixed-grid-native-route-execution'


def _check(condition, message):
    if not condition:
        raise ValueError('CPU fixed-grid native route execution: ' + message)


def execute_cpu_fixed_grid_native_route(package_ref, out, *, stable_inputs_for_process_lifetime=False):
    package = read_artifact(package_ref)
    resolved = prepare_cpu_fixed_grid_native_route_launch(package_ref, argv=package['argv'], cpu_ids=package['cpu_ids'],
        target=package['hardware'], environment=package['source_environment'],
        stable_inputs_for_process_lifetime=stable_inputs_for_process_lifetime)
    if resolved['status'] == 'DECLINE_NATIVE_ROUTE':
        return resolved

    output = Path(package['publication']['path'])
    out = Path(out).absolute()
    _check(out.resolve() == out and not out.is_relative_to(Path(package_ref['uri']).parent)
           and not any(out.is_relative_to(row['project_root']) for row in package['routes'].values())
           and not out.is_relative_to(output.parent), 'fresh receipt directory outside the generated projects and publication directory required')
    out.mkdir(parents=True, exist_ok=False)
    execution = resolved['execution']
    run = subprocess.run(execution['argv'], cwd=execution['cwd'], env=execution['environment'],
        stdin=subprocess.DEVNULL, capture_output=True, timeout=execution['timeout_seconds'])
    (out/'stdout').write_bytes(run.stdout)
    (out/'stderr').write_bytes(run.stderr)
    expected_mode = package['publication']['expected_mode']
    _check(run.returncode == 0 and run.stderr == b''
           and run.stdout == read_artifact(package['stdout'], json_document=False)
           and output.is_file() and not output.is_symlink() and stat.S_IMODE(output.stat().st_mode) == expected_mode,
           'selected route did not preserve the expected exit, output streams or file mode')
    actual = _ref(output, 'scar.native-float32-output')
    expected = package['publication']['expected_file']
    _check(all(actual[key] == expected[key] for key in ('sha256', 'size_bytes')), 'selected route changed the complete output bytes')
    archived = out/'published.f32'
    shutil.move(output, archived)
    report = dict(schema=SCHEMA, schema_version=1, status='SCOPED_NATIVE_ROUTE_EXECUTED', package=package_ref,
        source=package['source'], target=package['hardware'], cpu_ids=package['cpu_ids'], route=resolved['route'],
        executable=resolved['executable'], command=execution['argv'], working_directory=execution['cwd'],
        environment_sha256=package['environment_sha256'], returncode=run.returncode,
        stdout=_ref(out/'stdout', 'application/octet-stream'), stderr=_ref(out/'stderr', 'application/octet-stream'),
        publication=dict(file=_ref(archived, 'scar.native-float32-output'), source_path=str(output), mode=expected_mode),
        exact_exit_streams_file_and_mode_verified=True, benchmark=False, retry_performed=False,
        target_executed=True, deployment_authorized=False)
    _write(out/'result.json', report)
    return report

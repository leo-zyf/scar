"""Seal an explicitly chosen, already built fixed-grid C++20 source CLI."""

import errno
import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory

from scar.codegen.cpu_array_native_route_package import _pin
from scar.codegen.cpu_catalog_runtime_package import _host, _signature, _write_json
from scar.codegen.native_cpu_elf import cpu_executable_dynamic
from scar.codegen.native_provider_bundle import _content
from scar.compiler.aten_entry_delivery import LAUNCHER
from scar.compiler.entry_delivery import _environment_digest
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.cpu_fixed_grid_mapped_input_plan import SCHEMA as PLAN, verify_cpu_fixed_grid_mapped_input_plan


SCHEMA = 'scar.cpu-fixed-grid-native-route-package'
_PACKAGE = 'artifact.json'
_OWNED, _MAPPED = 'OWNED_COPY', 'PROVIDER_MAP'
_DEPENDENCIES = 'plan/dependencies.json'


def _need(condition, detail):
    if not condition:
        raise ValueError('CPU fixed-grid native route package: ' + detail)


def _ordinary_executable(reference):
    path = _pin(reference)
    _need(os.access(path, os.X_OK) and not path.stat().st_mode & (stat.S_ISUID | stat.S_ISGID),
          'ordinary executable permission required: ' + str(path))
    return path


def _route(reference, name, executable, launcher, plan):
    root = _pin(reference).parent
    project = read_artifact(reference)
    descriptor = project['source_entry_cli']
    binary = project['executable']
    _need(project['cxx_standard'] == 20 and descriptor['deployment_authorized'] is False
          and descriptor.get('mapped_input_proof') == (plan['proof'] if name == _MAPPED else None)
          and descriptor['launcher'] == LAUNCHER
          and executable['uri'] == str(root/binary) and launcher['uri'] == str(root/LAUNCHER)
          and _signature(executable) == project['files'][binary]
          and _signature(launcher) == project['files'][LAUNCHER], 'sealed native route differs from observed C++ project')
    _ordinary_executable(executable)
    _ordinary_executable(launcher)
    locks = dict(schema='scar.native-dependency-locks', uri=str(root/_DEPENDENCIES), **project['files'][_DEPENDENCIES])
    _pin(locks)
    return dict(project=reference, project_root=str(root), executable=executable, launcher=launcher,
        cpu_ids=project['cpus'], loader=dict(dynamic=cpu_executable_dynamic(executable['uri']),
                                            resolved=project['dependencies'], locks=locks))


def _derive(plan_ref):
    _need(plan_ref['schema'] == PLAN, 'exact fixed-grid input route plan required')
    plan = verify_cpu_fixed_grid_mapped_input_plan(plan_ref)
    _need(plan['observation'] is not None and plan['empirical'] is not None and plan['execution_default'] == 'NO_OP'
          and plan['transformation_authorized'] is False and plan['deployment_authorized'] is False
          and plan['whole_entry_equivalence_proven'] is False, 'declared native source scope with measured route required')
    study = read_artifact(plan['observation'])
    delivery_ref = study['control_delivery']
    delivery = read_artifact(delivery_ref)
    abi = read_artifact(read_artifact(plan['proof'])['abi'])
    argv, cpus, target = study['argv'], study['cpu_ids'], plan['target']
    slot = abi['argv']['output'] - 1
    cwd, env = study['working_directory'], study['environment']
    selected = plan['selected_runtime']
    _need(plan['scope']['distinct_fresh_output_required'] is True
          and plan['scope']['stable_inputs_required_for_full_lifetime'] is True
          and plan['project'] == study['control_project'] == delivery['project']
          and plan['target'] == study['target'] == delivery['target']
          and list(argv) == delivery['argv'] and len(argv) == abi['argv']['count'] == 3
          and slot == len(argv) - 1 and Path(argv[slot]).is_absolute()
          and cpus == delivery['cpu_ids'] == plan['placement']['cpus'] and cpus and cpus == sorted(set(cpus))
          and cwd == delivery['working_directory'] and Path(cwd).resolve(strict=True) == Path(cwd) and Path(cwd).is_dir()
          and _environment_digest(env) == study['environment_sha256'] == delivery['environment_sha256']
          and env == delivery['launch']['provider_environment'] and env['PATH'] == '/usr/bin:/bin'
          and not any(key.startswith('LD_') for key in env)
          and study['file_inputs'] == delivery['file_inputs']
          and study['original_stdout'] == delivery['validated_source_stdout']
          and study['original_output'] == delivery['validated_final_tensor']
          and selected in (_OWNED, _MAPPED)
          and plan['runtime_switch_eligible'] is (selected == _MAPPED),
          'route envelope differs from the exact observed stable-file native invocation')
    _need(study['control_executable'] == delivery['native_executable']
          and study['launchers'][_OWNED] == delivery['native_launcher'], 'owned-copy baseline differs from delivery')
    routes = {
        _OWNED: _route(study['control_project'], _OWNED, study['control_executable'], study['launchers'][_OWNED], plan),
        _MAPPED: _route(study['mapped_project'], _MAPPED, study['mapped_executable'], study['launchers'][_MAPPED], plan),
    }
    _need(plan['selected_project'] == routes[selected]['project']
          and routes[_OWNED]['cpu_ids'] == routes[_MAPPED]['cpu_ids'] == cpus
          and routes[_OWNED]['loader']['resolved'] == routes[_MAPPED]['loader']['resolved']
          and all(_content(Path(path)) == _signature(reference) for path, reference in delivery['launch_tools'].items()),
          'selected project, placement, shell or provider linkage differs')
    for reference in (study['original_stdout'], study['original_output']):
        _pin(reference)
    return dict(schema=SCHEMA, schema_version=1, policy='EXPLICIT_SCOPED_NATIVE_ROUTE_ONLY',
        status=plan['status'], plan=plan_ref, observation=plan['observation'], owned_copy_delivery=delivery_ref,
        selected_runtime=selected, selected_project=plan['selected_project'], routes=routes, host=_host(),
        source=study['source'], hardware=target, argv=argv, cpu_ids=cpus, steps=study['steps'],
        file_inputs=study['file_inputs'], storage=study['storage'], native_placement=study['native_placement'],
        source_environment=env, environment_sha256=study['environment_sha256'], working_directory=cwd,
        launch_tools=delivery['launch_tools'], stdout=study['original_stdout'], timeout_seconds=120,
        publication=dict(path=argv[slot], output_index=slot, expected_file=study['original_output'],
                         expected_mode=study['expected_mode'], precondition='ABSENT_CXX_EXCLUSIVE_CREATE'),
        inputs_must_remain_stable=True, cxx_standard=20, python_runtime_required=False,
        original_python_action='KEEP_ORIGINAL', python_replacement_authorized=False,
        native_fallback_after_execution=False, deployment_authorized=False, target_executed=False)


def export_cpu_fixed_grid_native_route_package(plan_ref, out):
    """Publish only the validated bindings to the existing C++20 projects."""
    package = _derive(plan_ref)
    out = Path(out).absolute()
    _need(out.resolve() == out and not out.exists() and not out.is_symlink()
          and all(not out.is_relative_to(route['project_root']) for route in package['routes'].values()),
          'fresh destination outside the retained native projects required')
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-fixed-grid-route-', dir=out.parent) as directory:
        stage = Path(directory)/'package'
        stage.mkdir(mode=0o700)
        _write_json(stage/_PACKAGE, package)
        _need(not out.exists() and not out.is_symlink(), 'route package destination became occupied')
        stage.rename(out)
    return artifact_reference(out/_PACKAGE, SCHEMA)


def _read_package(reference):
    _need(reference['schema'] == SCHEMA and _pin(reference).name == _PACKAGE,
          'exact fixed-grid native route package required')
    return read_artifact(reference)


def verify_cpu_fixed_grid_native_route_package(reference):
    """Recheck sealed program, scoped decision, loader and evidence without launching either arm."""
    package = _read_package(reference)
    _need(package == _derive(package['plan']), 'retained native route package differs at replay')
    return package


def _output_guards(output, expected_mode):
    parent = output.parent
    if output.exists() or output.is_symlink():
        return ['OUTPUT_ALREADY_OCCUPIED']
    if not parent.is_dir() or parent.resolve(strict=True) != parent:
        return ['EXACT_OUTPUT_PARENT_REQUIRED']
    try:
        default_acl = os.getxattr(parent, 'system.posix_acl_default', follow_symlinks=False)
    except OSError as error:
        if error.errno not in (errno.ENODATA, errno.ENOTSUP):
            return ['OUTPUT_PERMISSIONS_NOT_QUALIFIED']
        default_acl = None
    match = next((line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                  if line.startswith('Umask:')), None)
    if default_acl is not None or match is None or (0o666 & ~int(match, 8)) != expected_mode:
        return ['OUTPUT_PERMISSIONS_NOT_QUALIFIED']
    return []


def prepare_cpu_fixed_grid_native_route_launch(reference, *, argv, cpu_ids, target, environment,
                                              stable_inputs_for_process_lifetime=False):
    """Resolve a native-only invocation for the sealed bytes and fresh output."""
    package = _read_package(reference)
    _need(type(stable_inputs_for_process_lifetime) is bool, 'explicit input stability premise must be boolean')
    publication = package['publication']
    output = Path(publication['path'])
    reasons = [field + '_DOES_NOT_MATCH' for field, provided, expected in (
        ('ARGV', argv, package['argv']), ('COHORT', cpu_ids, package['cpu_ids']),
        ('TARGET', target, package['hardware']), ('ENVIRONMENT', environment, package['source_environment']),
    ) if provided != expected]
    if not set(package['cpu_ids']) <= os.sched_getaffinity(0):
        reasons.append('COHORT_NOT_AVAILABLE')
    if package['selected_runtime'] == _MAPPED and not stable_inputs_for_process_lifetime:
        reasons.append('FULL_INPUT_LIFETIME_STABILITY_REQUIRED')
    if not reasons:
        reasons = _output_guards(output, publication['expected_mode'])
    if not reasons:
        # Recheck after reading the complete plan and provider; C++ still uses O_EXCL.
        _need(package == _derive(package['plan']), 'retained native route package differs at replay')
        reasons = _output_guards(output, publication['expected_mode'])
    common = dict(original_python_action='KEEP_ORIGINAL', fallback_after_execution=False,
                  deployment_authorized=False, target_executed=False)
    if reasons:
        return dict(status='DECLINE_NATIVE_ROUTE', reasons=reasons, execution=None, **common)
    route = package['routes'][package['selected_runtime']]
    return dict(status='SCOPED_NATIVE_INVOCATION', route=package['selected_runtime'], executable=route['executable'],
        execution=dict(argv=[route['launcher']['uri'], *package['argv']], cwd=package['working_directory'],
                       environment=package['source_environment'], timeout_seconds=package['timeout_seconds']),
        publication=publication, stdout=package['stdout'], input_files=package['file_inputs'], **common)

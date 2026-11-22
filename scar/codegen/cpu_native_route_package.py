"""Bind scoped native route plans to retained C++20 projects by reference."""

import errno
import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory

from scar.analysis.replay_transaction import replay_transaction
from scar.codegen.cpu_catalog_runtime_package import _host, _loader, _signature, _write_json
from scar.codegen.native_cpu_arena_reuse import _path
from scar.codegen.native_provider_bundle import _content, cpu_executable_dynamic
from scar.compiler.aten_entry_delivery import LAUNCHER
from scar.compiler.entry_delivery import _environment_digest
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.cpu_fixed_grid_mapped_input_plan import SCHEMA as _FIXED_GRID_PLAN


ARRAY_SCHEMA = 'scar.cpu-array-native-route-package'
FIXED_GRID_SCHEMA = 'scar.cpu-fixed-grid-native-route-package'
_PACKAGE = 'artifact.json'
_ARRAY_PLAN = 'scar.cpu-array-native-route-plan'
_GROUPED = 'scar.native-cpu-fp-guard-groups'
_BASE = 'scar.native-seeded-array-task-entry'
_CHALLENGER, _INCUMBENT = 'grouped_openmp', 'retained_process'
_TASKSET = Path('/usr/bin/taskset')
_DIAGNOSTICS = {'SCAR_ARRAY_ENTRY_STATES', 'SCAR_ARRAY_ENTRY_LANE_REPORT'}
_GROUPED_MODE, _PROCESS_MODE = 'ORDERED_OPENMP', 'ORDERED_LOCAL_PROCESSES'
_OWNED, _MAPPED = 'OWNED_COPY', 'PROVIDER_MAP'
_DEPENDENCIES = 'plan/dependencies.json'


def _require_array(condition, reason):
    if not condition:
        raise ValueError('CPU array native route package: ' + reason)


def _require_fixed_grid(condition, detail):
    if not condition:
        raise ValueError('CPU fixed-grid native route package: ' + detail)


def _pin(reference, require):
    path = Path(reference['uri'])
    mode = path.lstat().st_mode
    require(path.resolve(strict=True) == path and stat.S_ISREG(mode), 'expected an ordinary pinned local file')
    require(_content(path) == _signature(reference), 'pinned file changed: ' + str(path))
    return path


def _publish_package(out, package, require, *, fresh_error, occupied_error):
    out = Path(out).absolute()
    require(out.resolve() == out and not out.exists() and not out.is_symlink()
            and all(not out.is_relative_to(route['project_root']) for route in package['routes'].values()),
            fresh_error)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-native-route-', dir=out.parent) as directory:
        stage = Path(directory)/'package'
        stage.mkdir(mode=0o700)
        _write_json(stage/_PACKAGE, package)
        require(not out.exists() and not out.is_symlink(), occupied_error)
        stage.rename(out)
    return artifact_reference(out/_PACKAGE, package['schema'])


def _read_package(reference, schema, require, detail):
    require(reference['schema'] == schema and _pin(reference, require).name == _PACKAGE, detail)
    return read_artifact(reference)


def _replay_package(package, derive, require, detail):
    require(package == derive(package['plan']), detail)
    return package


def _invocation_refusals(package, argv, cpu_ids, target, environment):
    reasons = [field + '_DOES_NOT_MATCH' for field, provided, expected in (
        ('ARGV', argv, package['argv']), ('COHORT', cpu_ids, package['cpu_ids']),
        ('TARGET', target, package['hardware']), ('ENVIRONMENT', environment, package['source_environment']),
    ) if provided != expected]
    if not set(package['cpu_ids']) <= os.sched_getaffinity(0):
        reasons.append('COHORT_NOT_AVAILABLE')
    return reasons


def _array_project(route, name, argv, cpus, environment, scope, recipe_ref):
    reference = route['package']
    grouped = name == _CHALLENGER
    _require_array(reference['schema'] == (_GROUPED if grouped else _BASE), 'unexpected native project schema')
    _pin(reference, _require_array)
    project = read_artifact(reference)
    root = Path(reference['uri']).parent
    binary = project['binary']
    _require_array(project['schema'] == reference['schema'] and project['cxx_standard'] == 20
             and project['deployment_authorized'] is False and project['python_runtime_required'] is False
             and project['recipe'] == recipe_ref and project['scope'] == scope and project['cpu_ids'] == cpus
             and {'CMakeLists.txt', 'src/main.cpp', binary} <= project['files'].keys()
             and project['files'][binary] == route['executable']
             and project['launch_environment'] == route['launch_environment']
             and not project.get('launch_prefix')
             and project['decision']['mode'] == (_GROUPED_MODE if grouped else _PROCESS_MODE),
             'native route crossed a pinned project, C++ version, profile or mode')
    for relative, member in project['files'].items():
        _require_array(_path(root, relative) == _pin(member, _require_array), 'native file escaped its compiled project')
    executable = _path(root, binary)
    _require_array(os.access(executable, os.X_OK) and not executable.stat().st_mode & (stat.S_ISUID | stat.S_ISGID),
             'ordinary executable required')
    launch_environment = dict(environment, **project['launch_environment'])
    _require_array(not (_DIAGNOSTICS | {'LD_PRELOAD', 'LD_AUDIT', 'LD_LIBRARY_PATH'}).intersection(launch_environment),
             'native clean launch has diagnostic or loader injection')
    members = {'project/' + name: _signature(member) for name, member in project['files'].items()}
    linkage = _loader(executable, root, members, str(root), launch_environment)
    _require_array(linkage['dynamic'] == project['linkage']['dynamic'], 'ELF dynamic tags differ from compilation')
    for soname, expected in project['linkage'].get('qualified_runtime_dependencies', {}).items():
        observed = linkage['dependencies'][soname]
        _require_array(observed['kind'] == 'HOST_FILE'
                 and Path(observed['path']).resolve(strict=True) == _pin(expected, _require_array)
                 and _signature(observed) == _signature(expected), 'qualified host library has changed: ' + soname)
    launcher = [str(_TASKSET), '-c', ','.join(map(str, cpus))]
    return dict(package=reference, project_root=str(root), executable=route['executable'], mode=project['decision']['mode'],
        launch=dict(command=[*launcher, str(executable), *argv], environment=launch_environment,
                    environment_sha256=_environment_digest(launch_environment)),
        loader=linkage)


def _derive_array(plan_ref):
    from scar.optimization.cpu_array_native_route_plan import verify_cpu_array_native_route_plan

    _require_array(plan_ref['schema'] == _ARRAY_PLAN, 'exact Stage 2 routing plan required')
    plan = verify_cpu_array_native_route_plan(plan_ref)
    _require_array(plan['execution_default'] == 'NO_OP' and plan['transformation_authorized'] is False
             and plan['deployment_authorized'] is False and plan['original_default_retained'] is True,
             'plan must expose only an explicit, scoped native route')
    selected, fallback = plan['selected_route'], plan['fallback_route']
    routes = plan['routes']
    _require_array(set(routes) == {_CHALLENGER, _INCUMBENT} and selected in routes and fallback == _INCUMBENT
             and routes[selected]['executable'] == plan['selected_executable']
             and routes[fallback]['executable'] == plan['fallback_executable'], 'plan route and binary IDs differ')
    workload, resources = plan['workload'], plan['resources']
    recipe_ref, holdout_ref = workload['recipe'], workload['holdout']
    recipe, holdout = read_artifact(recipe_ref), read_artifact(holdout_ref)
    argv, cpus, environment = holdout['argv'], holdout['cpu_ids'], recipe['environment']
    target = resources['hardware']
    read_artifact(target)
    flags, = recipe['options'][recipe['output_option']]['flags']
    _require_array(workload['argv'] == argv and holdout['recipe'] == recipe_ref and holdout['entry'] == recipe['entry']
             and holdout['environment'] == environment and holdout['normalized_runtime_fields'] == recipe['runtime_fields']
             and workload['source'] == recipe['entry'] and workload['normalized_oracle'] == holdout['normalized']
             and workload['original_environment'] == environment and workload['normalization'] == recipe['runtime_fields']
             and workload['output_index'] == len(argv) - 1
             and resources['launch_cpu_ids'] == cpus and cpus and cpus == sorted(set(cpus))
             and argv[-2:] == [flags, 'published.json'] and argv.count(flags) == 1,
             'original holdout, native argument scope or sole relative publication slot differs')
    for reference in (recipe['entry'], holdout['normalized']):
        _pin(reference, _require_array)
    sealed = {name: _array_project(route, name, argv, cpus, environment, plan['scope'], recipe_ref)
              for name, route in sorted(routes.items())}
    _require_array(sealed[_INCUMBENT]['launch']['environment'] == workload['local_process_environment']
             and sealed[_CHALLENGER]['launch']['environment'] == workload['openmp_grouped_environment'],
             'native launch environment differs from the respective tested route')
    return dict(schema=ARRAY_SCHEMA, schema_version=1, policy='EXPLICIT_SCOPED_NATIVE_ROUTE_ONLY',
        deployment_authorized=False, python_replacement_authorized=False, original_python_action='KEEP_ORIGINAL',
        cxx_standard=20, python_runtime_required=False, target_executed=False,
        plan=plan_ref, status=plan['status'], source=recipe['entry'], holdout=holdout_ref, recipe=recipe_ref,
        hardware=target, cpu_ids=cpus, scope=plan['scope'], host=_host(), selected_route=selected, fallback_route=fallback,
        argv=argv, output_slot=len(argv) - 1, output_leaf=argv[-1], oracle=holdout['normalized'],
        runtime_fields=recipe['runtime_fields'], source_environment=environment,
        original_working_directory=workload['cwd'],
        launcher=artifact_reference(_TASKSET, 'scar.native-launch-tool'), working_directory='FRESH_OUTPUT_DIRECTORY',
        timeout_seconds=120, routes=sealed)


def derive_cpu_array_native_route_package(plan_ref):
    """Read and bind an accepted native routing plan to exact C++20 projects."""
    with replay_transaction():
        return _derive_array(plan_ref)


def export_cpu_array_native_route_package(plan_ref, out):
    """Publish the verified launch bindings by reference and leave both projects in place."""
    return _publish_package(out, derive_cpu_array_native_route_package(plan_ref), _require_array,
                            fresh_error='fresh package outside both compiled projects required',
                            occupied_error='publication became occupied')


def _read_array_package(reference):
    return _read_package(reference, ARRAY_SCHEMA, _require_array, 'invalid route package reference')


def _replay_array_package(package):
    return _replay_package(package, derive_cpu_array_native_route_package, _require_array,
                           'sealed native route differs on replay')


def verify_cpu_array_native_route_package(reference):
    """Recompute the exact package and locate its dependencies without running an entry."""
    return _replay_array_package(_read_array_package(reference))


def _rebind_probed_package(reference, checked_package):
    """Reuse a checked launch when only the choice and its decision receipt changed."""
    checked_ref, checked = checked_package
    if checked_ref == reference:
        return checked
    original = read_artifact(reference)
    decision_fields = {'plan', 'status', 'selected_route', 'fallback_route'}
    _require_array(reference['schema'] == ARRAY_SCHEMA
             and read_artifact(original['plan'])['proof'] == read_artifact(checked['plan'])['proof']
             and {key: value for key, value in original.items() if key not in decision_fields}
             == {key: value for key, value in checked.items() if key not in decision_fields},
             'probed route package differs in binaries, resources, environment or output contract')
    return original


def prepare_cpu_array_native_route_launch(reference, *, argv, cpu_ids, target, environment, fresh_root,
                                          prefer_original_native=False):
    """Resolve an explicitly requested native invocation or decline before any target executes."""
    package = _read_array_package(reference)
    root = Path(fresh_root).absolute()
    package_root = Path(reference['uri']).parent
    _require_array(type(prefer_original_native) is bool, 'explicit native fallback flag must be boolean')
    reasons = _invocation_refusals(package, argv, cpu_ids, target, environment)
    if (root.resolve() != root or root.exists() or root.is_symlink() or root.is_relative_to(package_root)
            or any(root.is_relative_to(row['project_root']) for row in package['routes'].values())):
        reasons.append('FRESH_OUTPUT_DIRECTORY_REQUIRED')
    if reasons:
        return dict(status='DECLINE_NATIVE_ROUTE', reasons=reasons, execution=None,
                    original_python_action='KEEP_ORIGINAL', deployment_authorized=False, target_executed=False)
    _replay_array_package(package)
    name = package['fallback_route'] if prefer_original_native else package['selected_route']
    route = package['routes'][name]
    return dict(status='SCOPED_NATIVE_INVOCATION', route=name, executable=route['executable'],
        execution=dict(argv=route['launch']['command'], cwd=str(root), environment=route['launch']['environment'],
                       timeout_seconds=package['timeout_seconds'], create_cwd_exclusively=True),
        publication=str(root/package['output_leaf']), oracle=package['oracle'], runtime_fields=package['runtime_fields'],
        original_python_action='KEEP_ORIGINAL', deployment_authorized=False, target_executed=False)


def _ordinary_executable(reference):
    path = _pin(reference, _require_fixed_grid)
    _require_fixed_grid(os.access(path, os.X_OK) and not path.stat().st_mode & (stat.S_ISUID | stat.S_ISGID),
          'ordinary executable permission required: ' + str(path))
    return path


def _fixed_grid_route(reference, name, executable, launcher, plan):
    root = _pin(reference, _require_fixed_grid).parent
    project = read_artifact(reference)
    descriptor = project['source_entry_cli']
    binary = project['executable']
    _require_fixed_grid(project['cxx_standard'] == 20 and descriptor['deployment_authorized'] is False
          and descriptor.get('mapped_input_proof') == (plan['proof'] if name == _MAPPED else None)
          and descriptor['launcher'] == LAUNCHER
          and executable['uri'] == str(root/binary) and launcher['uri'] == str(root/LAUNCHER)
          and _signature(executable) == project['files'][binary]
          and _signature(launcher) == project['files'][LAUNCHER], 'sealed native route differs from observed C++ project')
    _ordinary_executable(executable)
    _ordinary_executable(launcher)
    locks = dict(schema='scar.native-dependency-locks', uri=str(root/_DEPENDENCIES), **project['files'][_DEPENDENCIES])
    _pin(locks, _require_fixed_grid)
    return dict(project=reference, project_root=str(root), executable=executable, launcher=launcher,
        cpu_ids=project['cpus'], loader=dict(dynamic=cpu_executable_dynamic(executable['uri']),
                                            resolved=project['dependencies'], locks=locks))


def _derive_fixed_grid(plan_ref):
    from scar.optimization.cpu_fixed_grid_mapped_input_plan import verify_cpu_fixed_grid_mapped_input_plan

    _require_fixed_grid(plan_ref['schema'] == _FIXED_GRID_PLAN, 'exact fixed-grid input route plan required')
    plan = verify_cpu_fixed_grid_mapped_input_plan(plan_ref)
    _require_fixed_grid(plan['observation'] is not None and plan['empirical'] is not None and plan['execution_default'] == 'NO_OP'
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
    _require_fixed_grid(plan['scope']['distinct_fresh_output_required'] is True
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
    _require_fixed_grid(study['control_executable'] == delivery['native_executable']
          and study['launchers'][_OWNED] == delivery['native_launcher'], 'owned-copy baseline differs from delivery')
    routes = {
        _OWNED: _fixed_grid_route(study['control_project'], _OWNED, study['control_executable'],
                                  study['launchers'][_OWNED], plan),
        _MAPPED: _fixed_grid_route(study['mapped_project'], _MAPPED, study['mapped_executable'],
                                   study['launchers'][_MAPPED], plan),
    }
    _require_fixed_grid(plan['selected_project'] == routes[selected]['project']
          and routes[_OWNED]['cpu_ids'] == routes[_MAPPED]['cpu_ids'] == cpus
          and routes[_OWNED]['loader']['resolved'] == routes[_MAPPED]['loader']['resolved']
          and all(_content(Path(path)) == _signature(reference) for path, reference in delivery['launch_tools'].items()),
          'selected project, placement, shell or provider linkage differs')
    for reference in (study['original_stdout'], study['original_output']):
        _pin(reference, _require_fixed_grid)
    return dict(schema=FIXED_GRID_SCHEMA, schema_version=1, policy='EXPLICIT_SCOPED_NATIVE_ROUTE_ONLY',
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
    return _publish_package(out, _derive_fixed_grid(plan_ref), _require_fixed_grid,
                            fresh_error='fresh destination outside the retained native projects required',
                            occupied_error='route package destination became occupied')


def _read_fixed_grid_package(reference):
    return _read_package(reference, FIXED_GRID_SCHEMA, _require_fixed_grid,
                         'exact fixed-grid native route package required')


def _replay_fixed_grid_package(package):
    return _replay_package(package, _derive_fixed_grid, _require_fixed_grid,
                           'retained native route package differs at replay')


def verify_cpu_fixed_grid_native_route_package(reference):
    """Recheck sealed program, scoped decision, loader and evidence without launching either arm."""
    return _replay_fixed_grid_package(_read_fixed_grid_package(reference))


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
    package = _read_fixed_grid_package(reference)
    _require_fixed_grid(type(stable_inputs_for_process_lifetime) is bool, 'explicit input stability premise must be boolean')
    publication = package['publication']
    output = Path(publication['path'])
    reasons = _invocation_refusals(package, argv, cpu_ids, target, environment)
    if package['selected_runtime'] == _MAPPED and not stable_inputs_for_process_lifetime:
        reasons.append('FULL_INPUT_LIFETIME_STABILITY_REQUIRED')
    if not reasons:
        reasons = _output_guards(output, publication['expected_mode'])
    if not reasons:
        # Recheck after reading the complete plan and provider; C++ still uses O_EXCL.
        _replay_fixed_grid_package(package)
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

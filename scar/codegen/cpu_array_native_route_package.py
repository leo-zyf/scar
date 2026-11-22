"""Seal an explicit native array-task route without copying either C++20 project."""

import json
import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory

from scar.analysis.replay_transaction import replay_transaction
from scar.codegen.cpu_catalog_runtime_package import _host, _loader, _signature, _write_json
from scar.codegen.native_cpu_arena_reuse import _path
from scar.codegen.native_provider_bundle import _content
from scar.compiler.entry_delivery import _environment_digest
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.cpu-array-native-route-package'
_PLAN = 'scar.cpu-array-native-route-plan'
_GROUPED = 'scar.native-cpu-fp-guard-groups'
_BASE = 'scar.native-seeded-array-task-entry'
_CHALLENGER, _INCUMBENT = 'grouped_openmp', 'retained_process'
_PACKAGE = 'artifact.json'
_TASKSET = Path('/usr/bin/taskset')
_DIAGNOSTICS = {'SCAR_ARRAY_ENTRY_STATES', 'SCAR_ARRAY_ENTRY_LANE_REPORT'}
_GROUPED_MODE, _PROCESS_MODE = 'ORDERED_OPENMP', 'ORDERED_LOCAL_PROCESSES'


def _require(condition, reason):
    if not condition:
        raise ValueError('CPU array native route package: ' + reason)


def _pin(reference):
    path = Path(reference['uri'])
    mode = path.lstat().st_mode
    _require(path.resolve(strict=True) == path and stat.S_ISREG(mode), 'expected an ordinary pinned local file')
    _require(_content(path) == _signature(reference), 'pinned file changed: ' + str(path))
    return path


def _project(route, name, argv, cpus, environment, scope, recipe_ref):
    reference = route['package']
    grouped = name == _CHALLENGER
    _require(reference['schema'] == (_GROUPED if grouped else _BASE), 'unexpected native project schema')
    _pin(reference)
    project = read_artifact(reference)
    root = Path(reference['uri']).parent
    binary = project['binary']
    _require(project['schema'] == reference['schema'] and project['cxx_standard'] == 20
             and project['deployment_authorized'] is False and project['python_runtime_required'] is False
             and project['recipe'] == recipe_ref and project['scope'] == scope and project['cpu_ids'] == cpus
             and {'CMakeLists.txt', 'src/main.cpp', binary} <= project['files'].keys()
             and project['files'][binary] == route['executable']
             and project['launch_environment'] == route['launch_environment']
             and not project.get('launch_prefix')
             and project['decision']['mode'] == (_GROUPED_MODE if grouped else _PROCESS_MODE),
             'native route crossed a pinned project, C++ version, profile or mode')
    for relative, member in project['files'].items():
        _require(_path(root, relative) == _pin(member), 'native file escaped its compiled project')
    executable = _path(root, binary)
    _require(os.access(executable, os.X_OK) and not executable.stat().st_mode & (stat.S_ISUID | stat.S_ISGID),
             'ordinary executable required')
    launch_environment = dict(environment, **project['launch_environment'])
    _require(not (_DIAGNOSTICS | {'LD_PRELOAD', 'LD_AUDIT', 'LD_LIBRARY_PATH'}).intersection(launch_environment),
             'native clean launch has diagnostic or loader injection')
    members = {'project/' + name: _signature(member) for name, member in project['files'].items()}
    linkage = _loader(executable, root, members, str(root), launch_environment)
    _require(linkage['dynamic'] == project['linkage']['dynamic'], 'ELF dynamic tags differ from compilation')
    for soname, expected in project['linkage'].get('qualified_runtime_dependencies', {}).items():
        observed = linkage['dependencies'][soname]
        _require(observed['kind'] == 'HOST_FILE' and Path(observed['path']).resolve(strict=True) == _pin(expected)
                 and _signature(observed) == _signature(expected), 'qualified host library has changed: ' + soname)
    launcher = [str(_TASKSET), '-c', ','.join(map(str, cpus))]
    return dict(package=reference, project_root=str(root), executable=route['executable'], mode=project['decision']['mode'],
        launch=dict(command=[*launcher, str(executable), *argv], environment=launch_environment,
                    environment_sha256=_environment_digest(launch_environment)),
        loader=linkage)


def _derive(plan_ref):
    from scar.optimization.cpu_array_native_route_plan import verify_cpu_array_native_route_plan

    _require(plan_ref['schema'] == _PLAN, 'exact Stage 2 routing plan required')
    plan = verify_cpu_array_native_route_plan(plan_ref)
    _require(plan['execution_default'] == 'NO_OP' and plan['transformation_authorized'] is False
             and plan['deployment_authorized'] is False and plan['original_default_retained'] is True,
             'plan must expose only an explicit, scoped native route')
    selected, fallback = plan['selected_route'], plan['fallback_route']
    routes = plan['routes']
    _require(set(routes) == {_CHALLENGER, _INCUMBENT} and selected in routes and fallback == _INCUMBENT
             and routes[selected]['executable'] == plan['selected_executable']
             and routes[fallback]['executable'] == plan['fallback_executable'], 'plan route and binary IDs differ')
    workload, resources = plan['workload'], plan['resources']
    recipe_ref, holdout_ref = workload['recipe'], workload['holdout']
    recipe, holdout = read_artifact(recipe_ref), read_artifact(holdout_ref)
    argv, cpus, environment = holdout['argv'], holdout['cpu_ids'], recipe['environment']
    target = resources['hardware']
    read_artifact(target)
    flags, = recipe['options'][recipe['output_option']]['flags']
    _require(workload['argv'] == argv and holdout['recipe'] == recipe_ref and holdout['entry'] == recipe['entry']
             and holdout['environment'] == environment and holdout['normalized_runtime_fields'] == recipe['runtime_fields']
             and workload['source'] == recipe['entry'] and workload['normalized_oracle'] == holdout['normalized']
             and workload['original_environment'] == environment and workload['normalization'] == recipe['runtime_fields']
             and workload['output_index'] == len(argv) - 1
             and resources['launch_cpu_ids'] == cpus and cpus and cpus == sorted(set(cpus))
             and argv[-2:] == [flags, 'published.json'] and argv.count(flags) == 1,
             'original holdout, native argument scope or sole relative publication slot differs')
    for reference in (recipe['entry'], holdout['normalized']):
        _pin(reference)
    sealed = {name: _project(route, name, argv, cpus, environment, plan['scope'], recipe_ref)
              for name, route in sorted(routes.items())}
    _require(sealed[_INCUMBENT]['launch']['environment'] == workload['local_process_environment']
             and sealed[_CHALLENGER]['launch']['environment'] == workload['openmp_grouped_environment'],
             'native launch environment differs from the respective tested route')
    return dict(schema=SCHEMA, schema_version=1, policy='EXPLICIT_SCOPED_NATIVE_ROUTE_ONLY',
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
        return _derive(plan_ref)


def export_cpu_array_native_route_package(plan_ref, out):
    """Publish the verified launch bindings by reference and leave both projects in place."""
    report = derive_cpu_array_native_route_package(plan_ref)
    out = Path(out).absolute()
    _require(out.resolve() == out and not out.exists() and not out.is_symlink()
             and all(not out.is_relative_to(row['project_root']) for row in report['routes'].values()),
             'fresh package outside both compiled projects required')
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-array-native-route-', dir=out.parent) as directory:
        stage = Path(directory)/'package'
        stage.mkdir(mode=0o700)
        _write_json(stage/_PACKAGE, report)
        _require(not out.exists() and not out.is_symlink(), 'publication became occupied')
        stage.rename(out)
    return artifact_reference(out/_PACKAGE, SCHEMA)


def _read_package(reference):
    _require(reference['schema'] == SCHEMA and _pin(reference).name == _PACKAGE, 'invalid route package reference')
    return read_artifact(reference)


def verify_cpu_array_native_route_package(reference):
    """Recompute the exact package and locate its dependencies without running an entry."""
    report = _read_package(reference)
    _require(report == derive_cpu_array_native_route_package(report['plan']), 'sealed native route differs on replay')
    return report


def _rebind_probed_package(reference, checked_package):
    """Reuse a checked launch when only the choice and its decision receipt changed."""
    checked_ref, checked = checked_package
    if checked_ref == reference:
        return checked
    original = read_artifact(reference)
    decision_fields = {'plan', 'status', 'selected_route', 'fallback_route'}
    _require(reference['schema'] == SCHEMA
             and read_artifact(original['plan'])['proof'] == read_artifact(checked['plan'])['proof']
             and {key: value for key, value in original.items() if key not in decision_fields}
             == {key: value for key, value in checked.items() if key not in decision_fields},
             'probed route package differs in binaries, resources, environment or output contract')
    return original


def prepare_cpu_array_native_route_launch(reference, *, argv, cpu_ids, target, environment, fresh_root,
                                          prefer_original_native=False):
    """Resolve an explicitly requested native invocation or decline before any target executes."""
    package = _read_package(reference)
    root = Path(fresh_root).absolute()
    package_root = Path(reference['uri']).parent
    _require(type(prefer_original_native) is bool, 'explicit native fallback flag must be boolean')
    reasons = []
    for field, observed, expected in (
        ('ARGV', argv, package['argv']), ('COHORT', cpu_ids, package['cpu_ids']),
        ('TARGET', target, package['hardware']), ('ENVIRONMENT', environment, package['source_environment']),
    ):
        if observed != expected:
            reasons.append(field + '_DOES_NOT_MATCH')
    if not set(package['cpu_ids']) <= os.sched_getaffinity(0):
        reasons.append('COHORT_NOT_AVAILABLE')
    if (root.resolve() != root or root.exists() or root.is_symlink() or root.is_relative_to(package_root)
            or any(root.is_relative_to(row['project_root']) for row in package['routes'].values())):
        reasons.append('FRESH_OUTPUT_DIRECTORY_REQUIRED')
    if reasons:
        return dict(status='DECLINE_NATIVE_ROUTE', reasons=reasons, execution=None,
                    original_python_action='KEEP_ORIGINAL', deployment_authorized=False, target_executed=False)
    _require(package == derive_cpu_array_native_route_package(package['plan']), 'sealed native route differs on replay')
    name = package['fallback_route'] if prefer_original_native else package['selected_route']
    route = package['routes'][name]
    return dict(status='SCOPED_NATIVE_INVOCATION', route=name, executable=route['executable'],
        execution=dict(argv=route['launch']['command'], cwd=str(root), environment=route['launch']['environment'],
                       timeout_seconds=package['timeout_seconds'], create_cwd_exclusively=True),
        publication=str(root/package['output_leaf']), oracle=package['oracle'], runtime_fields=package['runtime_fields'],
        original_python_action='KEEP_ORIGINAL', deployment_authorized=False, target_executed=False)

"""Cpu entry launch scope."""

from dataclasses import asdict
from hashlib import sha256
import json
import sys
import argparse
from pathlib import Path
from scar.analysis.cpu_stream_graph import _OBSERVER, _retained
from scar.analysis.scalar_recursive_driver import CAPTURE
from scar.compiler.cpu_entry_replay import SCHEMA as PAIRED_REPLAY
from scar.compiler.plan_aten_boundary import _target
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.analysis.build import build
from scar.analysis.cpu_import_source_body import (
    SCHEMA as BODY,
    derive_cpu_import_source_body,
    verify_cpu_import_source_body,
)
from scar.analysis.replay_transaction import detached_replay
from scar.compiler.frontend.python import _Run, build_frontend
from scar.analysis.static_program import read_static_program
from scar.optimization.cpu_entry_import_frontier import (
    SCHEMA as FRONTIER,
    verify_cpu_entry_import_frontier,
)
from scar.optimization.scope_accounting import _pointer


# Project original CPU launch requirements from opt-in retained evidence.
def _require(condition, message):
    if not condition:
        raise ValueError('retained CPU launch: ' + message)


def _context(interpreter, cpu_ids, cwd, environment):
    path = Path(interpreter)
    directory = Path(cwd)
    _require(path.is_absolute() and directory.is_absolute() and directory.is_dir()
             and directory.resolve(strict=True) == directory,
             'absolute original interpreter and canonical working directory required')
    path = path.resolve(strict=True)
    # These archive schemas do not record Python's version or cache tag. Recover
    # them only from the exact executable already hosting the compiler process.
    _require(path == Path(sys.executable).resolve(strict=True),
             'captured interpreter must match the compiler executable for the Python ABI')
    _require(type(cpu_ids) is list and bool(cpu_ids)
             and all(type(cpu) is int and cpu >= 0 for cpu in cpu_ids)
             and sorted(set(cpu_ids)) == cpu_ids, 'canonical original CPU cohort required')
    _require(type(environment) is dict and all(type(name) is str and name
             and '=' not in name and '\x00' not in name and type(value) is str and '\x00' not in value
             for name, value in environment.items()), 'invalid retained original environment')
    names = {'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'}
    names.update(name for name in environment if name.startswith(('OMP_', 'GOMP_', 'KMP_')))
    environment_sha256 = sha256(json.dumps(sorted(environment.items()), separators=(',', ':')).encode()).hexdigest()
    return dict(working_directory=str(directory), python_executable=str(path),
                python_version=sys.version, python_cache_tag=sys.implementation.cache_tag,
                process_affinity_cpu_ids=cpu_ids,
                native_thread_environment={name: environment.get(name) for name in sorted(names)},
                environment_sha256=environment_sha256)


def _paired(reference, entry, argv, stream_capture):
    report = read_artifact(reference)
    _require(reference['schema'] == PAIRED_REPLAY and report['schema_version'] == 1
             and report['capture'] == stream_capture,
             'paired observation and explicit stream capture differ')
    compilation = read_artifact(report['compilation'])
    spec = report['invocation']
    _require(compilation['entry'] == spec['entry'] == entry and compilation['argv'] == argv,
             'paired observation identifies another original entry or argv')
    original = spec['arms']['original']
    read_artifact(original['executable'], json_document=False)
    return _context(original['executable']['uri'], spec['cpu_ids'], spec['cwd'], original['environment'])


def verify_paired_original_scope(scope, reference):
    """Check exact original launch identity for evidence added after planning."""
    report = read_artifact(reference)
    spec = report['invocation']
    original = spec['arms']['original']
    compilation = read_artifact(report['compilation'])
    environment = sha256(json.dumps(sorted(original['environment'].items()), separators=(',', ':')).encode()).hexdigest()
    _require(reference['schema'] == PAIRED_REPLAY
             and compilation['entry'] == spec['entry']
             and (spec['entry']['uri'], spec['entry']['sha256']) == (scope['entry'], scope['entry_sha256'])
             and compilation['argv'] == scope['argv']
             and spec['cpu_ids'] == scope['process_affinity_cpu_ids']
             and spec['cwd'] == scope['working_directory']
             and str(Path(original['executable']['uri']).resolve(strict=True)) == scope['python_executable']
             and environment == scope['environment_sha256'],
             'paired evidence must match the original entry, argv, CPU cohort and environment')
    return report, compilation


def _stream(reference, scope, entry, paired):
    capture = read_artifact(reference)
    _require(reference['schema'] == CAPTURE and capture['source'] == entry,
             'stream capture identifies another original entry')
    launch = json.loads(read_artifact(capture['captured_launch'], json_document=False))
    command = launch['command']
    _require(type(command) is list and all(type(arg) is str and '\x00' not in arg for arg in command)
             and command.count(_OBSERVER) == 1, 'structured original observer launch required')
    observer = command.index(_OBSERVER)
    _require(observer >= 4, 'original observer must name an interpreter')
    if command[:2] == ['/usr/bin/taskset', '-c']:
        value = command[2]
        _require(len(value) <= 10 and value.isascii() and value.isdecimal() and value == str(int(value)),
                 'stream observer must bind one canonical CPU')
        cpus = [int(value)]
    else:
        _require(paired is not None and observer == 4,
                 'stream observer must bind one CPU or have a verified paired observation')
        cpus = paired['process_affinity_cpu_ids']
    delta = _context(command[observer - 4], cpus, launch['cwd'], launch['environment'])
    _require(paired is None or delta == paired, 'paired measurement and stream observer launch differ')
    _retained(reference, dict(scope, **delta, cpu_stream_capture=reference))
    return delta


def project_cpu_entry_launch_scope(scope, *, stream_capture=None, paired_observation=None):
    """Return launch requirements for later mandatory F4 and provider replay."""
    _require(stream_capture is not None or paired_observation is not None,
             'retained original launch authority required')
    entry = artifact_reference(Path(scope['entry']).resolve(strict=True), 'scar.python-source')
    _require(entry['uri'] == scope['entry'] and entry['sha256'] == scope['entry_sha256'],
             'original F4 source identity changed')
    delta = None if paired_observation is None else _paired(paired_observation, entry, scope['argv'], stream_capture)
    if stream_capture is not None:
        delta = _stream(stream_capture, scope, entry, delta)
    cpus = delta['process_affinity_cpu_ids']
    target, _ = _target(Path(scope['target']['uri']).parent, cpus, allow_cpu_subset=True)
    _require(asdict(target) == scope['target'], 'original CPU cohort must belong to the exact F4 target')
    return delta


# Build exact observed child module sources through the local F4 frontend.
SCHEMA = 'scar.cpu-import-source-admission'
_PROGRAM = 'scar.program-build'
_SOURCE = 'scar.python-source'
_FRONTEND = 'scar.source-frontend'
_CATALOG = 'scar.declared-source-catalog'
_MODEL = 'scar.static-program'
_TARGET = 'scar.target-hardware-graph'


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU import source admission: ' + detail)


def _ref(path, schema):
    path = Path(path).absolute()
    _need(path.resolve(strict=True) == path and path.is_file(), 'expected a canonical retained file')
    return artifact_reference(path, schema)


def _converted(row, schema):
    return dict(uri=row['path'], sha256=row['sha256'], size_bytes=row['bytes'], schema=schema)


def _write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def _inputs(reference):
    frontier = verify_cpu_entry_import_frontier(reference)
    _need(frontier['schema_version'] == 2 and frontier['target_executed'] is False,
          'expected a process-scoped v2 child import frontier')
    program = read_artifact(frontier['program'])
    target = _converted(program['target'], _TARGET)
    read_artifact(target)
    bindings = read_artifact(frontier['bindings'])
    candidates, sources = [], {}
    for index, candidate in enumerate(frontier['source_admission_candidates']):
        source = _pointer(bindings, candidate['source'])
        path = Path(source['uri'])
        _need(candidate['status'] == 'SOURCE_ADMISSION_CANDIDATE' and source['schema'] == _SOURCE
              and path.suffix == '.py' and _ref(path, _SOURCE) == source,
              'candidate is outside its exact observed Python source')
        previous = sources.setdefault(path, source)
        _need(previous == source, 'candidate source versions disagree at one pathname')
        candidates.append(dict(module_name=candidate['module_name'], original_import=candidate['original_import'],
            source=source, candidate=f'#/source_admission_candidates/{index}'))
    return frontier, target, candidates, dict(sorted(sources.items()))


def _frontend(root, reference, sources):
    if not sources:
        _need(reference is None, 'empty source admission has no frontend')
        return None
    paths = list(map(str, sources))
    source_directory = root/'source-model'
    model_ref = _ref(source_directory/'manifest.json', _MODEL)
    frontend_path = root/'frontend/result.json'
    _need(reference == _ref(frontend_path, _FRONTEND), 'frontend must belong to the exact archived selection')
    frontend = read_artifact(reference)
    _need(frontend['selected_sources'] == paths and frontend['trace'] is None
          and frontend['target_executed'] is False and frontend['workflow_complete'] is True
          and frontend['local_declarations_complete'] is True and _converted(frontend['model'], _MODEL) == model_ref,
          'exact-file frontend did not finish: ' + str(frontend.get('error', 'incomplete receipt')))
    model = read_static_program(source_directory, verify=False)
    _need(model['phase'] == 'sources' and model['target_executed'] is False
          and model['source_selection']['directory_roots'] == [str(root/'empty-source-root')]
          and model['source_selection']['explicit_files'] == paths
          and model['source_catalog_status'] == 'COMPLETE' and not model['worker_errors']
          and not model['source_changed'] and len(model['units']) == len(sources)
          and {row['path']: (row['raw_sha256'], row['source_bytes']) for row in model['units']}
          == {str(path): (ref['sha256'], ref['size_bytes']) for path, ref in sources.items()},
          'source model contains unobserved files or changed bytes')
    catalog = _converted(frontend['catalog'], _CATALOG)
    _need(catalog == _ref(root/'frontend/catalog/result.json', _CATALOG), 'exact frontend catalog changed')
    return catalog


def _body(program_ref, source, target, catalog, value):
    program = read_artifact(program_ref)
    _need(program_ref['schema'] == _PROGRAM and program['backend'] == 'F4' and program['trace'] is None
          and program['target_executed'] is False and program['workflow_complete'] is True
          and program['summary']['configuration_inputs'] == 0 and program['entry'] == source['uri']
          and _converted(program['f4']['catalog'], _CATALOG) == catalog
          and _converted(program['target'], _TARGET) == target and value['target'] == target
          and value['source'] == source and value['program'] == program_ref and value['target_executed'] is False,
          'initializer body differs from its exact original target, catalog or source')


def _report(reference, frontier, target, candidates, frontend_ref, body_refs, body_modules):
    modules = []
    for row in candidates:
        path = Path(row['source']['uri'])
        _need(body_modules[path] == row['module_name'], 'observed name differs from the archived F4 module identity')
        modules.append(dict(row, body=body_refs[path]))
    return dict(schema=SCHEMA, schema_version=1, frontier=reference, observation=frontier['observation'],
        entry_obligations=frontier['entry_obligations'], program=frontier['program'], target=target,
        modules=modules, frontend=frontend_ref, unique_source_count=len(body_refs),
        scope='EXACT_CHILD_MODULE_SOURCE_FILES_ONLY', runtime_initializer_activation_proven=False,
        import_effects_closed=False, target_executed=False, deployment_authorized=False)


def _source_model(root, sources, jobs):
    empty = root/'empty-source-root'
    empty.mkdir()
    model = root/'source-model'
    layers = root/'source-layers'
    layers.mkdir()
    report = layers/'result.json'
    files = [item for path in sources for item in ('--source-file', path)]
    _Run(layers).stage('source-layers', 'scar.compiler.frontend.source',
        [empty, *files, '--model', model, '--report', report, '--jobs', jobs,
         '--max-files', len(sources), '--max-bytes', sum(row['size_bytes'] for row in sources.values()),
         '--max-directories', 1], report, 'scar.source-layer-batch',
        complete_for_declared_units=True, target_executed=False)
    return model


def build_cpu_import_source_admission(reference, out, *, jobs=4):
    """Build a shared exact-file catalog and one source-only F4 program per module file."""
    _need(type(jobs) is int and 1 <= jobs <= 4, 'worker budget must be in [1, 4]')
    frontier, target, candidates, sources = _inputs(reference)
    root = Path(out).absolute()
    _need(root.resolve() == root and not root.exists() and all(not path.is_relative_to(root) for path in sources),
          'fresh canonical output outside original source paths required')
    root.mkdir(parents=True, exist_ok=False)
    frontend_ref, bodies, names = None, {}, {}
    if sources:
        model = _source_model(root, sources, jobs)
        roots = tuple(dict.fromkeys(path.parent for path in sources))
        frontend_path = root/'frontend'
        build_frontend(roots, frontend_path, sources=tuple(sources), source_model=model, jobs=jobs)
        frontend_ref = _ref(frontend_path/'result.json', _FRONTEND)
        catalog = _frontend(root, frontend_ref, sources)
        for ordinal, (path, source) in enumerate(sources.items()):
            folder = root/f'module-{ordinal:04d}'
            build(path, None, folder, jobs=jobs, target=Path(target['uri']).parent,
                  catalog=Path(catalog['uri']).parent, cache_roots=(frontend_path,), configuration_roots=())
            program = _ref(folder/'manifest.json', _PROGRAM)
            body = derive_cpu_import_source_body(program, source)
            _body(program, source, target, catalog, body)
            _write(folder/'body.json', body)
            bodies[path] = _ref(folder/'body.json', BODY)
            names[path] = body['module']
    result = _report(reference, frontier, target, candidates, frontend_ref, bodies, names)
    _write(root/'result.json', result)
    return result


def verify_cpu_import_source_admission(reference):
    """Replay the original evidence and every generated initializer body, without rebuilding."""
    identity = tuple(reference[key] for key in ('schema', 'uri', 'sha256', 'size_bytes'))
    return detached_replay('scar.cpu-import-source-admission.v1', identity, lambda: _verify(reference))


def _verify(reference):
    saved = read_artifact(reference)
    path = Path(reference['uri'])
    _need(reference['schema'] == SCHEMA and saved['schema_version'] == 1 and path.name == 'result.json'
          and path.resolve(strict=True) == path, 'exact retained source admission required')
    root = path.parent
    frontier_ref = saved['frontier']
    frontier, target, candidates, sources = _inputs(frontier_ref)
    catalog = _frontend(root, saved['frontend'], sources)
    bodies, names = {}, {}
    for ordinal, (source_path, source) in enumerate(sources.items()):
        folder = root/f'module-{ordinal:04d}'
        program = _ref(folder/'manifest.json', _PROGRAM)
        body_ref = _ref(folder/'body.json', BODY)
        body = verify_cpu_import_source_body(body_ref)
        _body(program, source, target, catalog, body)
        bodies[source_path] = body_ref
        names[source_path] = body['module']
    rebuilt = _report(frontier_ref, frontier, target, candidates, saved['frontend'], bodies, names)
    _need(saved == rebuilt, 'retained sources or module-body relationships differ on cold replay')
    return rebuilt


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar admit-cpu-import-sources', description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--frontier', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--jobs', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args(argv)
    if args.frontier is not None and args.out is not None:
        report = build_cpu_import_source_admission(_ref(args.frontier, FRONTIER), args.out, jobs=args.jobs)
        reference = _ref(args.out/'result.json', SCHEMA)
    elif args.check is not None and args.out is None:
        reference = _ref(args.check, SCHEMA)
        report = verify_cpu_import_source_admission(reference)
    else:
        parser.error('use --frontier and --out, or --check')
    print(json.dumps(dict(reference=reference, modules=len(report['modules']), target_executed=False), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Build exact observed child module sources through the local F4 frontend."""

import argparse
import json
from pathlib import Path

from scar.analysis.build import build
from scar.analysis.cpu_import_source_body import (
    SCHEMA as BODY, derive_cpu_import_source_body, verify_cpu_import_source_body,
)
from scar.analysis.replay_transaction import detached_replay
from scar.compiler.frontend.python import _LAYERS, _Run, build_frontend
from scar.analysis.static_program import build_static_program, read_static_program
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.cpu_entry_import_frontier import SCHEMA as FRONTIER, verify_cpu_entry_import_frontier
from scar.optimization.scope_accounting import _pointer


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
    build_static_program(empty, source_files=tuple(sources), out=model, phase='sources', jobs=1,
        max_files=len(sources), max_bytes=sum(row['size_bytes'] for row in sources.values()), max_directories=1)
    layers = root/'source-layers'
    layers.mkdir()
    run = _Run(layers)
    for command, schema in _LAYERS:
        out = layers/command
        run.worker(command, [command, '--model', model, '--out', out, '--jobs', jobs],
                  out/'result.json', schema, complete_for_declared_units=True)
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

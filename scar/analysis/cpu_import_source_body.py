"""Project one observed Python module's original F4 initializer source body."""

from collections import Counter
from pathlib import Path

from scar.analysis.build import BuildSession
from scar.analysis.cpu_entry_import_bindings import SOURCE, _source
from scar.analysis.entry_obligations import LIMITS, _Projection, _check_session
from scar.analysis.replay_transaction import detached_replay, replay_transaction
from scar.ir.execution_recipe import read_artifact
from scar.ir.record_codec import encode


SCHEMA = 'scar.f4-cpu-import-source-body'
_TARGET = 'scar.target-hardware-graph'
_LIMITS = dict(LIMITS, bodies=1)


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU import source body: ' + detail)


def _identity(reference):
    return tuple(reference[key] for key in ('schema', 'uri', 'sha256', 'size_bytes'))


def _project(program, source):
    with BuildSession(Path(program['uri']).parent) as session:
        _check_session(session, program)
        manifest = session.report
        _need(manifest['target_executed'] is False and manifest['trace'] is None,
              'a source-only F4 build is required')
        receipt = manifest['target']
        _need(type(receipt) is dict, 'the original target hardware graph must be pinned')
        target = dict(schema=_TARGET, uri=receipt['path'], sha256=receipt['sha256'], size_bytes=receipt['bytes'])
        read_artifact(target)

        view = session.view
        unit = manifest['f4']['entry_unit']
        store = view.catalog.fact_store
        expected = _source(store, unit)
        _need(source['schema'] == SOURCE and source == expected and source['uri'] == manifest['entry'],
              'observed source differs from the original archived F0 entry bytes')
        body = view.unit(unit)
        scope = body.facts.root_scope
        module = store.rows[unit]['module']
        _need(type(module) is str and module and body.facts.module == module
              and scope.owner.unit == unit and view.source_region_ref(
                  unit, scope.template, scope.region, context=()) == session.entry,
              'entry is not the source catalog module initializer')

        graph = _Projection(session, _LIMITS).run(session.entry)
        root, = graph['bodies']
        _need(graph['entry_body'] == 0 and root['point'] == encode(session.entry)
              and graph['units'][root['unit']]['unit'] == unit
              and all(option['body'] in (None, 0) for field in ('calls', 'imports')
                      for row in graph[field] for option in row['alternatives']),
              'initializer projection crossed its one-body boundary')
        store.validate_parents()
    return target, module, graph, root['status']


def derive_cpu_import_source_body(program_ref, source_ref):
    """Keep the original graph points and source-only effects for one initializer."""
    identity = _identity(program_ref), _identity(source_ref)
    with replay_transaction():
        target, module, graph, status = detached_replay('scar.cpu-import-source-body.v1', identity,
            lambda: _project(program_ref, source_ref))
    obligations = Counter(effect for op in graph['operations'] for effect in op['obligations'])
    summary = {name: len(graph[name]) for name in ('operations', 'edges', 'calls', 'imports')}
    summary['effect_obligations'] = dict(sorted(obligations.items()))
    return dict(schema=SCHEMA, schema_version=1, program=program_ref, source=source_ref, target=target,
                module=module, status='SOURCE_INITIALIZER_' + status, summary=summary,
                limits=dict(_LIMITS), projection=graph, scope='ORIGINAL_NORMAL_SUCCESS_MODULE_INITIALIZER_MAY_GRAPH',
                coverage_frontier=['RUNTIME_IMPORTER_SELECTION_CACHE_AND_ACTIVATION_OPEN',
                    'CALLEE_AND_NESTED_IMPORT_BODIES_OUTSIDE_PROJECTION',
                    'PROVIDER_ADDRESS_EXCEPTION_AND_EXTERNAL_EFFECTS_OPEN'],
                initializer_activation_proven=False, runtime_path_complete=False, effects_closed=False,
                target_executed=False, transformation_authorized=False, deployment_authorized=False)


def verify_cpu_import_source_body(body_ref):
    """Cold-replay the bounded module projection from its exact build and source."""
    saved = read_artifact(body_ref)
    _need(body_ref['schema'] == SCHEMA and saved == derive_cpu_import_source_body(saved['program'], saved['source']),
          'saved initializer body differs from original F4 replay')
    return saved

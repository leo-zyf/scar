"""Join an observed IMPORT_NAME to the original F4 request and module bodies."""

from collections import defaultdict
from pathlib import Path

from scar.analysis.build import BuildSession
from scar.analysis.cpu_entry_import_initializers import (
    SCHEMA as INITIALIZERS, _row, verify_cpu_entry_import_initializers,
)
from scar.analysis.replay_transaction import detached_replay, replay_transaction
from scar.analysis.source_bytes import read_source_bytes
from scar.analysis.source_decoded_bodies import reuse_decoded_source_bodies, source_body_access
from scar.analysis.source_productions import _digest
from scar.analysis.static_query import _Fragments
from scar.ir.execution_recipe import read_artifact
from scar.ir.literals import PythonLiteral
from scar.ir.program import normal_graph as G, source_productions as P
from scar.ir.program.source_template_links import SourceTemplateKey
from scar.ir.record_codec import decode, encode


SCHEMA = 'scar.f4-cpu-entry-import-bindings'
OBSERVATION = 'scar.cpu-original-import-bindings'
SOURCE = 'scar.python-source'
_POSITION = ('lineno', 'end_lineno', 'col_offset', 'end_col_offset')
_MATCHED = 'SOURCE_BODY_MATCHED'
_UNIQUE = 'UNIQUE_F4_IMPORT_REQUEST'


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU entry import bindings: ' + detail)


def _identity(reference):
    return tuple(reference[key] for key in ('uri', 'sha256', 'size_bytes'))


def _source(store, unit):
    if unit in store.original_source_units or unit not in store.model_units:
        return None
    row = store.model_units[unit]
    _need(Path(row['path']).is_absolute(), 'original source path is not absolute')
    archived = _Fragments(store.model_for(unit)).read(row['artifacts']['source_bytes'])
    _need(archived['path'] == row['path'] and archived['raw']['sha256'] == row['raw_sha256']
          and len(read_source_bytes(archived)) == row['source_bytes'], 'catalog row differs from original F0 source')
    return dict(uri=row['path'], sha256=row['raw_sha256'], size_bytes=row['source_bytes'], schema=SOURCE)


def _span(document, occurrence):
    node = _row(document['nodes'], occurrence)
    kind = node['constructor']
    _need(kind in ('Import', 'ImportFrom'), 'typed import is not its original statement occurrence')
    attributes, = (row['attributes'] for row in document['profile']['ast_schema'] if row['name'] == kind)
    values = dict(zip(attributes, node['attributes'], strict=True))
    result = []
    for name in _POSITION:
        slot = values.get(name)
        _need(slot is not None and slot['kind'] == 'SCALAR', 'import statement lacks original coordinates')
        number = decode(PythonLiteral, slot['scalar']).to_python()
        _need(type(number) is int and number >= 0, 'import coordinate is not an exact integer')
        result.append(number)
    return result


def _sites(session, original):
    store = session.view.catalog.fact_store
    requested = defaultdict(list)
    for ordinal, row in enumerate(original['imports']):
        candidates = [(i, alt) for i, alt in enumerate(row['alternatives'])
                      if alt['body'] is not None and _row(original['bodies'], alt['body'])['status'] == 'PROJECTED']
        operation = _row(original['operations'], row['operation'])
        port = _row(original['ports'], operation['reach_port'])
        owner = _row(original['owners'], port['owner'])
        template = _row(original['templates'], owner['template'])
        _need(port['kind'] == G.PortKind.ENTER.value and operation['unit'] == template['unit']
              and operation['template'] == template['template'], 'original import operation port differs')
        unit = _row(original['units'], operation['unit'])['unit']
        requested[unit].append((ordinal, candidates, operation, owner, decode(SourceTemplateKey, template['template'])))
    sites = []
    for unit, rows in requested.items():
        source_ref = _source(store, unit)
        if source_ref is None:
            # A compiled context can assign an arbitrary co_filename.
            continue
        source = session.view.unit(unit)
        model = store.model_units[unit]
        syntax = _Fragments(store.model_for(unit)).read(model['artifacts']['source_syntax'])
        _need(syntax['schema'] == 'scar.source-syntax' and syntax['status'] == 'PARSED'
              and _digest(syntax) == source.graph.input.f2_sha256
              and syntax['input']['path'] == source_ref['uri'] == source.graph.input.path
              and syntax['input']['raw_sha256'] == source_ref['sha256'], 'F2 receipt does not qualify F4 source')
        plans = source_body_access(source.material).plans(key for _, _, _, _, key in rows)
        for ordinal, candidates, operation, owner, key in rows:
            step, = (step for step in plans[key].steps if step.id.wire == owner['original'])
            request = step.instruction
            row = original['imports'][ordinal]
            _need(type(request) is P.ImportRequest and step.region.wire == operation['region']
                  and source._node(key, step.id).node.wire == operation['operation']
                  and (request.requested_name, request.relative_level, encode(request.from_names))
                  == (row['requested_name'], row['relative_level'], row['from_names']),
                  'typed import request differs from the retained F4 operation')
            alternatives = []
            for index, option in candidates:
                body = original['bodies'][option['body']]
                identity = _row(original['units'], body['unit'])['unit']
                expected = _source(store, identity)
                if expected is None:
                    continue
                alternatives.append(dict(name=option['module'], body=f"#/bodies/{option['body']}",
                    alternative=f'#/imports/{ordinal}/alternatives/{index}', source=expected))
            sites.append(dict(ordinal=ordinal, operation=row['operation'], file=source_ref,
                span=_span(syntax, key.occurrence), requested_name=request.requested_name,
                level=request.relative_level, fromlist=list(request.from_names or ()), alternatives=alternatives))
        source.material.validate_parents()
    return sorted(sites, key=lambda site: site['ordinal'])


@reuse_decoded_source_bodies()
def _resolve_static(initializer_ref):
    retained = read_artifact(initializer_ref)
    _need(initializer_ref['schema'] == INITIALIZERS, 'retained initializer graph required')
    original = read_artifact(retained['entry_obligations'])
    with BuildSession(Path(retained['program']['uri']).parent) as session:
        verified = verify_cpu_entry_import_initializers(initializer_ref, session=session)
        _need(verified['program'] == original['program'] and session.report == read_artifact(verified['program']),
              'initializer graph and original F4 build differ')
        body = _row(original['bodies'], original['entry_body'])
        unit = _row(original['units'], body['unit'])['unit']
        source = _source(session.view.catalog.fact_store, unit)
        _need(source is not None and source['uri'] == session.report['entry'],
              'entry does not have one original Python source snapshot')
        sites = _sites(session, original)
        session.view.catalog.fact_store.validate_parents()
    return verified, sites, source


def _static(initializer_ref):
    identity = (initializer_ref['schema'], *_identity(initializer_ref))
    return detached_replay('scar.cpu-entry-import-static.v1', identity, lambda: _resolve_static(initializer_ref))


def derive_cpu_entry_import_watch(initializer_ref):
    """Return unique F2 spans of original typed import requests."""
    _, sites, _ = _static(initializer_ref)
    records = {}
    for site in sites:
        key = _identity(site['file']), tuple(site['span'])
        row = records.setdefault(key, dict(file=site['file'], span=site['span'], imports=[]))
        row['imports'].append(site['ordinal'])
    return list(records.values())


def _key(path, span, name, level, fromlist):
    return path, tuple(span), name, level, tuple(fromlist)


def _module(observation, row, alternatives):
    candidates = [option for option in alternatives if option['name'] == row['name'] and row['source'] is not None
                  and row['origin'] == option['source']['uri']
                  and _identity(row['source']) == _identity(option['source'])
                  and row['activation'] in ('NEW_TO_CACHE', 'CACHED', 'REPLACED_IN_CACHE')]
    selected = candidates[0] if len(candidates) == 1 else None
    if row['activation'] == 'ABSENT':
        status = 'MODULE_CACHE_ABSENT'
    elif selected is not None:
        status = _MATCHED
    elif candidates:
        status = 'SOURCE_BODY_AMBIGUOUS'
    else:
        status = 'SOURCE_BODY_OPEN'
    return dict(observation=observation, name=row['name'], origin=row['origin'],
                origin_kind=row.get('origin_kind', 'UNKNOWN'), activation=row['activation'],
                source=row['source'], status=status, candidate_bodies=list(dict.fromkeys(c['body'] for c in candidates)),
                original_body=selected['body'] if selected else None,
                original_alternative=selected['alternative'] if selected else None)


def derive_cpu_entry_import_bindings(initializer_ref, observation_ref):
    """Bind only observed request/registry records and exact original source identities."""
    from scar.compiler.cpu_import_binding_observation import verify_cpu_import_binding_observation

    _need(observation_ref['schema'] == OBSERVATION, 'original import binding observation required')
    with replay_transaction():
        observed = verify_cpu_import_binding_observation(observation_ref)
        _need(observed['initializer_graph'] == initializer_ref, 'observation uses a different initializer graph')
        initializers, sites, source = _static(initializer_ref)
        pair = read_artifact(observed['paired'])
        _need(_identity(pair['invocation']['entry']) == _identity(source), 'paired source differs from the original F4 entry')

    process_scoped = observed['schema_version'] == 2
    process_slots = {row['slot'] for row in observed['processes'] if row['status'] == 'OBSERVED'} if process_scoped else None
    entries = defaultdict(list)
    for site in sites:
        entries[_key(site['file']['uri'], site['span'], site['requested_name'], site['level'], site['fromlist'])].append(site)
    bindings, matched = [], set()
    for index, row in enumerate(observed['entries']):
        pointer = f'#/entries/{index}'
        alternatives = entries[_key(row['caller'], row['position'], row['requested_name'], row['level'], row['fromlist'])]
        selected = alternatives[0] if len(alternatives) == 1 else None
        candidates = [f"#/imports/{site['ordinal']}" for site in alternatives]
        if selected:
            matched.add(selected['ordinal'])
        modules = [_module(f'{pointer}/modules/{i}', module, selected['alternatives'] if selected else ())
                   for i, module in enumerate(row['modules'])]
        binding = dict(observation=pointer, original_import=candidates[0] if selected else None,
            operation=selected['operation'] if selected else None, candidate_imports=candidates,
            status=_UNIQUE if selected else 'AMBIGUOUS_F4_IMPORT_REQUEST' if alternatives else 'UNMATCHED_F4_IMPORT_REQUEST',
            modules=modules)
        if process_scoped:
            _need(row['process_slot'] in process_slots, 'import request has no observed process instance')
            binding['process_slot'] = row['process_slot']
        bindings.append(binding)
    watched = {f"#/imports/{site['ordinal']}" for site in sites}
    positive = any(module['status'] == _MATCHED for binding in bindings for module in binding['modules'])
    report = dict(schema=SCHEMA, schema_version=2 if process_scoped else 1,
                initializer_graph=initializer_ref, observation=observation_ref,
                entry_obligations=initializers['entry_obligations'], program=initializers['program'], paired=observed['paired'],
                status='OBSERVED_SOURCE_IMPORT_BINDINGS' if positive else 'SOURCE_IMPORT_BINDINGS_OPEN',
                bindings=bindings,
                unobserved_imports=[f"#/imports/{site['ordinal']}" for site in sites if site['ordinal'] not in matched],
                unwatchable_imports=[row['original'] for row in initializers['requests'] if row['original'] not in watched],
                module_source_basis='ARCHIVED_F0_BYTES_AND_POST_RETURN_ORIGIN_PATH_READBACK',
                module_cache_identity_is_not_initializer_activation=True, initializer_execution_proven=False,
                runtime_import_effects_closed=False, whole_entry_effect_equivalence_proven=False,
                transformation_authorized=False, target_executed=False)
    if process_scoped:
        report.update(processes=observed['processes'], process_basis='VERIFIED_DIAGNOSTIC_SPAWN_LINEAGE',
                      process_execution_completeness_proven=False)
    return report


def verify_cpu_entry_import_bindings(reference):
    """Replay the exact archived observation and F4 route without executing the target."""
    saved = read_artifact(reference)
    _need(reference['schema'] == SCHEMA
          and saved == derive_cpu_entry_import_bindings(saved['initializer_graph'], saved['observation']),
          'saved import binding graph differs from replay')
    return saved

"""Publish and replay the source/target boundary of one original F4 build."""

from collections import Counter
from pathlib import Path

from scar.analysis.build import BuildSession
from scar.analysis.build_f4 import _validate_catalog_parents
from scar.analysis.replay_transaction import current_transaction, replay_transaction
from scar.artifacts import _durable_bytes
from scar.artifacts import _json
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.record_codec import encode


SCHEMA = 'scar.f4-build-admission'
PROGRAM = 'scar.program-build'
TARGET = 'scar.target-hardware-graph'


def _pin(reference):
    transaction = current_transaction()
    transaction.watch(reference['uri'], reference['sha256'], reference['size_bytes'])
    return reference


def _reference(receipt, schema):
    return _pin(dict(uri=receipt['path'], sha256=receipt['sha256'],
        size_bytes=receipt['bytes'], schema=schema))


def _source(store, unit, path):
    original = store.original_source_units.get(unit, unit)
    model = store.model_units[original]
    if Path(model['path']) != path or Path(store.rows[unit]['path']) != path:
        raise ValueError('F4 source unit differs from the catalog path: ' + str(path))
    return dict(unit=unit, original_unit=original, source=_reference(
        dict(path=str(path), sha256=model['raw_sha256'], bytes=model['source_bytes']),
        'scar.python-source-bytes'))


def _admit(session, program, hardware, source_filter):
    if (session.report.get('backend') != 'F4' or session.report.get('workflow_complete') is not True
            or program['schema'] != PROGRAM or Path(program['uri']) != session.path/'manifest.json'
            or read_artifact(program) != session.report):
        raise ValueError('a published original F4 build is required')
    if hardware is None or hardware != session.report.get('target'):
        raise ValueError('hardware receipt differs from the original F4 build target')

    spec = session.report['f4']
    catalog = session.view.catalog
    catalog_ref = _reference(spec['catalog'], catalog.report['schema'])
    if catalog.path/'result.json' != Path(catalog_ref['uri']) or read_artifact(catalog_ref) != catalog.report:
        raise ValueError('admitted source catalog differs from the original F4 build')
    store = catalog.fact_store
    entry_unit = spec['entry_unit']
    entry = _source(store, entry_unit, Path(session.report['entry']))
    if session.entry != session.view.entry(entry_unit, context=()):
        raise ValueError('admitted entry point differs from the F4 source namespace')
    entry['point'] = encode(session.entry)
    selected = []
    if source_filter is not None:
        source_filter = Path(source_filter).absolute()
        selected = [_source(store, unit, source_filter) for unit, row in sorted(store.rows.items())
            if Path(row['path']) == source_filter]
        if not selected:
            raise ValueError('planning source filter is outside the admitted F4 catalog: ' + str(source_filter))

    resources = session.resources
    target_ref = _reference(hardware, TARGET)
    if resources.target is None or encode(resources.target_ref) != target_ref:
        raise ValueError('joined resource interfaces differ from the original F4 target')
    target = resources.target
    storage = session.report['tables']
    gaps_ref = _reference(storage['gaps.json'], 'scar.build-gaps')
    gaps = read_artifact(gaps_ref)['gaps']
    query = _reference(storage['entry-query.json'], 'scar.program-dependency-query')
    observations = _reference(storage['observed.json'], 'scar.build-f4-observed-selections')
    live_observations = session.attach_trace()
    # Originals carry full endpoints, qualifiers, and evidence. Keep only
    # reference and count information at this boundary between stages.
    observed_calls = live_observations['bounded_call_bindings']
    metadata = dict(
        provider_names=sorted(session.session.providers),
        bounded_call_record_sets=len(observed_calls),
        bounded_call_attempts=sum(row['attempts'] for row in observed_calls),
        bounded_call_target_records=sum(row['target_records'] for row in observed_calls),
        call_target_metadata_proves_invocation_completion=False)
    availability = Counter((row.discovery_status, row.visibility_status, row.accessibility_status,
        row.backend_status) for row in target.availability())

    _validate_catalog_parents(catalog)
    resources.check()
    return dict(schema=SCHEMA, schema_version=1, program=program,
        source=dict(catalog=catalog_ref, entry=entry, catalog_units=len(store.rows),
            filter_path=None if source_filter is None else str(source_filter), filter_units=selected,
            unit_scope='CATALOG_DECLARATIONS_WITH_SEPARATE_BOUNDED_OBSERVATIONS'),
        target=dict(snapshot=target_ref, inventory=_pin(dict(target.report['inventory'],
                schema='scar.target-inventory')),
            resources_by_type=dict(Counter(type(row).__name__ for row in resources.target_resources)),
            connections=len(resources.connections), path_endpoints=len(resources.paths),
            availability=[dict(discovery=key[0], visibility=key[1], accessibility=key[2], backend=key[3],
                count=count) for key, count in sorted(availability.items())],
            process_constraint_records=len(target.constraints()),
            topology_proves_placement=False, inventory_proves_allocation_budget=False),
        evidence=dict(observed=observations, entry_query=query, **metadata),
        frontier=dict(saved_gaps=gaps_ref, saved_by_layer=dict(Counter(row['layer'] for row in gaps)),
            saved_by_kind=dict(Counter(row['kind'] for row in gaps)),
            whole_program_reachability_complete=False, runtime_path_complete=False,
            effects_closed=False, transformation_authorized=False))


def publish_build_admission(session, out_file, *, source_filter=None):
    """Publish a compact original F4 admission from an open planning session."""
    out_file = Path(out_file).absolute()
    with replay_transaction():
        program = _pin(artifact_reference(session.path/'manifest.json', PROGRAM))
        document = _admit(session, program, session.report.get('target'), source_filter)
    _durable_bytes(out_file, _json(document), replace=False)
    return artifact_reference(out_file, SCHEMA)


def verify_build_admission(reference, *, program, hardware, source_filter=None):
    """Replay independently from original typed F4 source and hardware tables."""
    with replay_transaction():
        saved = read_artifact(_pin(reference))
        if (reference['schema'] != SCHEMA or hardware is None
                or saved.get('program') != program
                or saved.get('target', {}).get('snapshot') != _reference(hardware, TARGET)):
            raise ValueError('F4 admission does not bind the original build and target')
        read_artifact(_pin(program))
        with BuildSession(Path(program['uri']).parent) as session:
            expected = _admit(session, program, hardware, source_filter)
        if _json(expected) != _json(saved):
            raise ValueError('F4 admission differs from original source/target replay')
    return saved

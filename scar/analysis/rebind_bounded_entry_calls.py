"""Reuse checked source correspondence and bind a new local CALL capture.

The prior projection is an explicit content-pinned proof input. Its source
routes retain their original conditional meaning. This distinct evidence
document carries fresh attempt identities and keeps unmatched targets open.
"""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from scar.analysis import bounded_entry_calls as projection
from scar.analysis.build import _rules
from scar.analysis.program_view import _digest
from scar.analysis.source_call_binding_view import SCHEMA as ATTEMPTS, SourceObservedCallTarget
from scar.analysis.source_declared_catalog import SCHEMA as DECLARED, read_source_catalog
from scar.compiler.call_binding import join_call_bindings, plan_digest, read_call_binding_plan
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program.ids import ProviderID
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import decode, encode

SCHEMA = 'scar.rebound-bounded-entry-calls'
RUN = 'scar.call-binding-run'
_PLAN = 'scar.call-binding-plan'
_RAW = 'scar.call-binding-observations'
_REF_FIELDS = {'uri', 'schema', 'size_bytes', 'sha256'}
_RESOURCE_SCOPE_READERS = frozenset((
    'source_resource_scopes.py', 'source_resource_scope_view.py', '../ir/program/resource_scopes.py',
))


class ReuseMiss(ValueError):
    """The prior bounded source proof cannot serve this capture."""


def _require(condition, reason):
    if not condition:
        raise ReuseMiss(reason)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class _Materials:
    """Deduplicate immutable file receipts and check the original read interval."""

    def __init__(self):
        self.refs = {}
        self.stamps = {}

    @staticmethod
    def _stamp(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def add(self, path, digest, size=None):
        path = Path(path)
        previous = self.refs.get(path)
        if previous is not None:
            _require(previous['sha256'] == digest and (size is None or size == previous['bytes']),
                     'conflicting static material receipts: ' + str(path))
            return
        _require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
                 'static material is not a canonical original file: ' + str(path))
        stamp = self._stamp(path)
        _require(size is None or size == stamp[2], 'static material extent differs: ' + str(path))
        raw = path.read_bytes()
        _require(sha256(raw).hexdigest() == digest and stamp == self._stamp(path),
                 'static material changed: ' + str(path))
        self.refs[path] = {'path': str(path), 'sha256': digest, 'bytes': len(raw)}
        self.stamps[path] = stamp

    def receipt(self, row):
        self.add(row['path'], row['sha256'], row.get('bytes'))

    def member(self, root, row):
        relative = Path(row['path'])
        _require(not relative.is_absolute() and '..' not in relative.parts,
                 'static material fragment escapes its original directory')
        self.add(root / relative, row['sha256'], row['bytes'])

    def snapshot(self):
        _require(all(self._stamp(path) == stamp for path, stamp in self.stamps.items()),
                 'static material changed during entry proof reuse')
        return [self.refs[path] for path in sorted(self.refs)]


def _artifacts(value):
    if isinstance(value, dict):
        if set(value) == _REF_FIELDS:
            yield value
        else:
            for item in value.values():
                yield from _artifacts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _artifacts(item)


def _templates(materials, root, units):
    from scar.analysis.source_template_links import _implementation, _worker_stores
    from scar.ir.program import source_template_links as link_types

    report = json.loads((root/'result.json').read_bytes())
    _require(report['schema'] == 'scar.source-template-link-store' and report['schema_version'] == 1
             and report['status'] == 'SOURCE_INTERFACES_LINKED'
             and report['generator_sha256'] == _implementation()
             and report['link_types_sha256'] == sha256(Path(link_types.__file__).read_bytes()).hexdigest(),
             'source template format or compiler profile changed')
    specifications = tuple((row['domain'], row['materials'], row['verification'], row['collection_sha256'],
                            row['verification_sha256']) for row in report['parents'])
    parents = _worker_stores(specifications, report['model_manifest_sha256'])
    index = {row['unit']: row for row in report['units']}
    for unit in units:
        row = index[unit]
        _require(row['header'] == f'units/{unit}/header.json', 'source template unit header differs')
        path = root/row['header']
        materials.add(path, row['header_sha256'], row['header_bytes'])
        header = json.loads(path.read_bytes())
        for table in ('templates', 'links'):
            materials.member(path.parent, header[table])
        for parent in parents:
            materials.add(parent.root/'result.json', parent.receipt['collection_sha256'])
            materials.add(parent.verification/'result.json', parent.receipt['verification_sha256'])
            materials.member(parent.root, parent.rows[unit]['artifact'])


def _static_materials(graph, program, plan):
    spec = program['f4']
    catalog_ref = spec['catalog']
    recorded, current = program['rules'], _rules()
    # F4 installs observed resource-scope readers only for an explicit
    # component. Source-only bounded CALL correspondence uses no such reader.
    inactive = (_RESOURCE_SCOPE_READERS if 'resource_scopes' not in spec['components'] else ())
    _require(plan['catalog'] == catalog_ref and program['trace'] is None
             and recorded.keys() == current.keys()
             and all(recorded[name] == digest for name, digest in current.items() if name not in inactive),
             'F4 catalog, reader rules or source-only projection profile changed')
    _require(artifact_reference(projection.__file__, 'scar.python-producer') == graph['producer'],
             'bounded source projection semantics changed')
    catalog = read_source_catalog(Path(catalog_ref['path']).parent)
    _require(catalog.report['schema'] == DECLARED, 'reuse supports declared source catalogs')
    store = catalog.fact_store
    materials = _Materials()
    materials.receipt(catalog_ref)
    for row in (*spec['functions'], *spec['components'].values()):
        materials.receipt(row)
    if program['target'] is not None:
        materials.receipt(program['target'])

    templates = defaultdict(list)
    for unit, row in store.rows.items():
        base = store.base
        parent = base.parents[base.unit_parent[unit]] if base.parents else base
        model = store.model_units[unit]
        materials.add(Path(model['path']), model['raw_sha256'])
        for artifact in model['artifacts'].values():
            materials.member(Path(store.model_for(unit)), artifact)
        materials.member(parent.path, {'path': row['facts'], 'sha256': row['sha256'], 'bytes': row['bytes']})
        materials.add(parent.templates/'result.json', parent.report['source_result_sha256'])
        templates[parent.templates].append(unit)
        local = catalog.interface_store.rows[unit]
        materials.member(catalog.interface_store.path,
                         {'path': local['interfaces'], 'sha256': local['sha256'], 'bytes': local['bytes']})
        for required in local['inputs']:
            materials.receipt(required)
    for directory, units in templates.items():
        _templates(materials, directory, units)
    catalog.validate_parents()

    # A logical source fact reference may resolve to a primitive shard in a
    # different collection. Its physical digest was checked above.
    from scar.analysis.source_fact_materials import FACTS, FactMaterialResolver
    from scar.ir.program.providers import ProviderFactReference

    resolver = FactMaterialResolver(catalog)
    static = {key: value for key, value in graph.items() if key != 'program'}
    unique_refs = {(ref['uri'], ref['size_bytes'], ref['sha256'], ref['schema']): ref for ref in _artifacts(static)}
    for ref in unique_refs.values():
        if ref['schema'] in (RUN, _RAW, _PLAN):
            continue
        if ref['schema'] == FACTS:
            route = resolver.route(ProviderFactReference(decode(ProviderArtifact, ref), '#/unit'))
            ref = encode(route.physical)
        materials.add(ref['uri'], ref['sha256'], ref['size_bytes'])
    return materials.snapshot()


def _run(reference):
    run = read_artifact(reference)
    _require(reference['schema'] == RUN and run['schema_version'] == 1
             and run['full_run_completed'] is True and run['transformations_disabled'] is True
             and run['child_returncode'] == 0 and run['stop_reason'] is None
             and len(run['raw_observations']) == 1, 'reuse requires one completed original bounded capture')
    folder = Path(reference['uri']).parent
    _require(Path(run['plan']) == folder/'plan.json' and Path(run['join']) == folder/'join.json'
             and Path(run['raw_observations'][0]).parent == folder, 'capture members escaped their original directory')
    plan_ref = artifact_reference(run['plan'], _PLAN)
    plan = read_call_binding_plan(plan_ref['uri'])
    raw_path = Path(run['raw_observations'][0])
    _require(raw_path.resolve(strict=True) == raw_path and raw_path.is_file()
             and 0 < raw_path.stat().st_size <= 2_000_000, 'raw capture member is not a bounded original file')
    raw_ref = artifact_reference(raw_path, _RAW)
    raw = read_artifact(raw_ref)
    joined = join_call_bindings(plan, raw)
    collector = raw['collector']
    _require(not raw['errors'] and collector['closed'] is True and collector['audit_hook_installed'] is True
             and collector['local_event'] == 'CALL' and collector['global_events_enabled'] is False
             and collector['profile_hook_installed'] is False
             and all(run[field] == joined[field] for field in ('planned_sites', 'observed_sites', 'attempts'))
             and read_artifact(artifact_reference(run['join'], joined['schema'])) == joined,
             'raw bounded capture or its original join differs')
    return plan, plan_ref, raw, raw_ref


def _old_matches(graph, raw, raw_ref, plan):
    planned = {site['id']: site for site in plan['sites']}
    known = defaultdict(list)
    seen = set()
    for row in graph['observed_targets']:
        observation = row['observation']
        record = decode(SourceObservedCallTarget, observation)
        evidence = observation['evidence']
        prefix, index = evidence['locator'].rsplit('/', 1)
        _require(prefix == '#/records' and index.isdecimal() and int(index) not in seen
                 and evidence['artifact'] == raw_ref, 'prior proof contains a foreign or duplicate attempt')
        ordinal = int(index)
        seen.add(ordinal)
        runtime = raw['records'][ordinal]
        site = planned[row['site_id']]
        _require((runtime['site_id'], row['unit'], row['query'], observation['caller_site']) ==
                 (site['id'], site['unit'], site['query'], site['site'])
                 and (record.process_id, record.thread_id, record.attempt_index, record.run) ==
                 (raw['process_id'], runtime['thread_id'], runtime['index'], raw_ref['sha256'])
                 and observation['target_code'] == runtime['target']['code']
                 and record.target_kind == runtime['target']['kind']
                 and record.metadata_json == _json(runtime['target']),
                 'prior observed target does not bind its original static site and raw attempt')
        known[site['id']].append(row)
    _require(len(seen) == len(raw['records']), 'prior bounded proof omits original attempts')
    return dict(known)


def _fresh(row, raw, raw_ref, ordinal, plan_hash):
    runtime = raw['records'][ordinal]
    locator = f'#/records/{ordinal}'
    provider = decode(ProviderArtifact, raw_ref)
    identity = ProviderID(_digest((ATTEMPTS, plan_hash, provider, locator)))
    observation = dict(row['observation'], id=encode(identity), run=raw_ref['sha256'],
                       process_id=raw['process_id'], thread_id=runtime['thread_id'],
                       attempt_index=runtime['index'], evidence={'artifact': raw_ref, 'locator': locator})
    decode(SourceObservedCallTarget, observation)
    return dict(row, observation=observation)


def rebind(prior, capture, *, required_sites=None):
    """Join a fresh capture to a known content-pinned original static proof.

    By default the complete bounded dispatch signature must recur. A consumer
    may instead require particular site IDs whose conditional source proof it
    has selected. Any unmatched outside attempts stay in the frontier.
    """
    graph = read_artifact(prior)
    _require(prior['schema'] == projection.SCHEMA and graph['schema_version'] == 1,
             'reuse requires an original bounded entry proof')
    program = read_artifact(graph['program'])
    _require(program['backend'] == 'F4' and program['workflow_complete'] is True
             and len(program['bounded_call_bindings']) == 1, 'prior proof lacks one complete published F4 channel')
    channel = program['bounded_call_bindings'][0]
    receipt = channel['run']
    old_run = {'uri': receipt['path'], 'sha256': receipt['sha256'], 'size_bytes': receipt['bytes'], 'schema': RUN}
    old_plan, old_plan_ref, old_raw, old_raw_ref = _run(old_run)
    plan, plan_ref, raw, raw_ref = _run(capture)
    _require(plan == old_plan and raw_ref != old_raw_ref and raw_ref['sha256'] != old_raw_ref['sha256'],
             'static call selectors changed or capture is the historical run')
    _require(channel['plan'] == {'path': old_plan_ref['uri'], 'bytes': old_plan_ref['size_bytes'],
                                 'sha256': old_plan_ref['sha256']}
             and channel['raw_observations'] == [{'path': old_raw_ref['uri'], 'bytes': old_raw_ref['size_bytes'],
                                                   'sha256': old_raw_ref['sha256']}],
             'prior F4 channel differs from its retained source proof')
    old = _old_matches(graph, old_raw, old_raw_ref, plan)
    new = defaultdict(list)
    for ordinal, row in enumerate(raw['records']):
        new[row['site_id']].append((ordinal, _json(row['target'])))
    planned = {site['id']: site for site in plan['sites']}
    required = set(planned) if required_sites is None else set(required_sites)
    _require(required and required <= planned.keys(), 'required site is outside the checked source plan')
    targets, frontiers, fingerprint = [], [], {}
    plan_hash = plan_digest(plan)
    for site in planned:
        original = old.get(site, [])
        actual = new.get(site, [])
        known = [row['observation']['metadata_json'] for row in original]
        if known != [metadata for _, metadata in actual]:
            _require(site not in required, 'observed target protocol or CodeID changed at ' + site)
            frontiers.append({'site_id': site, 'reason': 'NO_MATCHING_STATIC_TARGET_PROOF',
                              'attempts': [f'#/records/{index}' for index, _ in actual]})
            continue
        if site in required and required_sites is not None:
            _require(bool(actual), 'required consumer site has no fresh target observation: ' + site)
        fingerprint[site] = [dict(row, observation={key: row['observation'][key] for key in
            ('source_call', 'target_kind', 'target_code', 'metadata_json', 'remaining_fields')})
            for row in original]
        targets.extend(_fresh(row, raw, raw_ref, index, plan_hash)
                       for row, (index, _) in zip(original, actual, strict=True))

    materials = _static_materials(graph, program, plan)
    static_key = sha256(_json({'schema': SCHEMA, 'entry': program['entry'], 'entry_scope': graph['entry_scope'],
        'producer': graph['producer'], 'rules': program['rules'], 'material': materials,
        'plan': plan_hash, 'target_protocol': fingerprint}).encode()).hexdigest()
    return dict(schema=SCHEMA, schema_version=1, static_projection=prior, capture=capture, call_plan=plan_ref,
        raw_attempts=raw_ref, required_sites=None if required_sites is None else sorted(required),
        static_proof_key='entry-call-static-v1:' + static_key, static_material_count=len(materials),
        observed_targets=targets, frontiers=frontiers, summary=dict(
            planned_sites=len(plan['sites']), observed_attempts=len(raw['records']),
            rebound_attempts=len(targets), unmatched_sites=len(frontiers),
            observed_target_kinds=dict(Counter(row['observation']['target_kind'] for row in targets))),
        qualification=dict(whole_entry_graph_rebuilt=False, prior_runtime_observations_transferred=False,
            dispatch_proven_for_other_runs=False, first_stage_complete=False, transformation_authorized=False))


def check(reference):
    """Recreate the fresh bindings under the original static and raw receipts."""
    stored = read_artifact(reference)
    _require(reference['schema'] == SCHEMA, 'unexpected rebound evidence schema')
    expected = rebind(stored['static_projection'], stored['capture'], required_sites=stored['required_sites'])
    _require(expected == stored, 'stored rebound source proof differs from independent replay')
    return expected

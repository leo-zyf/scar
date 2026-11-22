"""Solve observed binding tracks over previously published static state rows."""

from collections import Counter
from hashlib import sha256
from itertools import chain
from pathlib import Path
from time import monotonic

from scar.analysis.binding_track_storage import ENCODING, TrackPacker, pins as storage_pins
from scar.artifacts import write_table
from scar.analysis.program_state_view import _edges, _pins, _tracks
from scar.analysis.program_view_shard_reader import open_source_graph_shard
from scar.analysis.source_state import local_read_tracks
from scar.artifacts import _durable_bytes, _durable_mkdir
from scar.artifacts import read_record_table as _table
from scar.artifacts import _json
from scar.analysis.source_value_flow import _artifact
from scar.ir.program import source_productions as P, state_ports as S
from scar.ir.record_codec import _decode, _record_tags, encode, loads


_STATIC_COUNTS = ('operations', 'binding_accesses', 'content_accesses', 'content_addresses',
    'content_states', 'content_relations', 'content_lineage_records', 'content_lineage_relations',
    'missing_address_rules')
_STATIC_TABLES = ('rules', 'requests', 'accesses', 'content_addresses', 'content_states',
    'content_accesses', 'content_edges', 'content_lineage', 'content_lineage_edges')


class BindingInventory:
    """Consume one admitted static shard while leaving the source recipes authoritative."""

    def __init__(self, state, inventory, index_sha256):
        self.state = state
        self.shard = shard = open_source_graph_shard(inventory, state.material.unit, index_sha256=index_sha256)
        header_path = Path(shard.unit_header.uri)
        raw = header_path.read_bytes()
        if sha256(raw).hexdigest() != shard.unit_header.sha256:
            raise ValueError('static binding shard changed after header admission')
        self.folder = header_path.parent / 'state'
        header = loads(raw)
        self.tables = tables = header['state_tables']
        self.counts = counts = header['counts']
        catalog = state.view.catalog
        if (shard.catalog != _artifact(catalog.path / 'result.json', catalog.report['schema'])
                or shard.source_material != state.material.result_artifact or shard.namespace != state.namespace
                or shard.namespace.graph != state.namespace.graph
                or set(tables) != {*_STATIC_TABLES, 'locations', 'edges', 'versions', 'selections', 'tracks'}
                or any(tables[n]['records'] != 0 for n in ('versions', 'selections', 'tracks'))
                or any(counts[n] != 0 for n in ('versions', 'read_selections', 'tracks'))
                or any(row['path'] != name + '.jsonl.gz' for name, row in tables.items())):
            raise ValueError('static binding shard uses a different graph or analysis contract')
        self.tags = _record_tags(P.PRODUCTION_RECORD_TAGS)

    def _records(self, name, cls):
        self._path(name)
        return (_decode(cls, row, self.tags) for row in _table(self.folder, self.tables[name]))

    def _path(self, name):
        path = self.folder / self.tables[name]['path']
        if path.resolve(strict=True) != path:
            raise ValueError('stored binding table must retain its original path: ' + name)
        return path

    def tracks(self, functions):
        """Load the static binding index and select the existing solver's inputs."""
        state = self.state
        if state.requests or state.accesses or state.locations or state.tracks or state.content_provider is not None:
            raise ValueError('static binding input requires a fresh source state provider')
        accesses = list(self._records('accesses', S.BindingAccessTemplate))
        locations = {row.id: row for row in self._records('locations', S.BindingLocationTemplate)}
        requests = {row.operation: row for row in self._records('requests', S.SourceEffectRequest)}
        counts = self.counts
        missing = Counter(row.rule.instruction for row in requests.values() if row.missing_rules)
        if ((len(accesses), len(locations), len(requests)) !=
                (counts['binding_accesses'], counts['locations'], counts['operations'])
                or any(len(rows) != self.tables[name]['records'] for name, rows in
                       (('accesses', accesses), ('locations', locations), ('requests', requests)))
                or dict(missing) != counts['missing_address_rules']
                or any(row.unit != state.material.unit for row in locations.values())
                or sorted(i for row in requests.values() for i in row.accesses) != list(range(len(accesses)))
                or any(access.operation not in requests or i not in requests[access.operation].accesses
                    or any(alt.location not in locations for alt in access.alternatives)
                    for i, access in enumerate(accesses))):
            raise ValueError('stored binding accesses, requests and locations do not close')
        state.locations.update(locations)
        state.accesses.extend(accesses)
        state.requests.update(requests)
        state.analyzed_operations.update(requests)
        state.bindings_complete = True
        state.check()
        selected = frozenset(functions)
        return tuple(pair for pair in local_read_tracks(state) if pair[0] in selected)

    def publish(self, out):
        """Publish the existing state-view wire with unchanged static tables."""
        started = monotonic()
        state = self.state
        if not state.bindings_complete or state.content_provider is not None:
            raise ValueError('static binding output requires the completed binding-only overlay')
        _durable_mkdir(out)
        tables = {}
        for name in _STATIC_TABLES:
            receipt = self.tables[name]
            data = self._path(name).read_bytes()
            if (len(data), sha256(data).hexdigest()) != (receipt['bytes'], receipt['sha256']):
                raise ValueError('static source state table changed: ' + name)
            _durable_bytes(out / receipt['path'], data, replace=False)
            tables[name] = receipt
        packer = TrackPacker()
        self._path('edges')
        updates = dict(locations=state.locations.values(), versions=state.versions.values(),
            selections=state.selections, tracks=(packer.pack(row) for row in _tracks(state)),
            controls=packer.controls, edges=chain(_table(self.folder, self.tables['edges']), _edges(state)))
        table_seconds = {}
        for name, records in updates.items():
            tables[name] = write_table(out / (name + '.jsonl.gz'), records, timings=table_seconds)
        counts = {name: self.counts[name] for name in _STATIC_COUNTS}
        counts.update(locations=len(state.locations), versions=len(state.versions),
            tracks=len(state.tracks), read_selections=len(state.selections), relations=tables['edges']['records'])
        if (len(state.requests), len(state.accesses), tables['edges']['records']) != (
                counts['operations'], counts['binding_accesses'],
                self.counts['relations'] + sum(len(rows) for rows in state.out_edges.values())):
            raise ValueError('binding solve changed its static input domain')
        catalog = state.view.catalog
        report = dict(schema='scar.source-state-view', schema_version=2, catalog=str(catalog.path),
            parent=encode(self.shard.catalog),
            unit=state.material.unit, namespace=encode(state.namespace), source_material=encode(state.material.result_artifact),
            binding_materials=encode(state.materials), pins=_pins(), storage_pins=storage_pins(), track_encoding=ENCODING,
            tables=tables, table_seconds=table_seconds, counts=counts, original_body_bytes_copied=0,
            loops_unrolled=False, target_executed=False, effect_membership_closed=False,
            tensor_content_versions_proven=False, first_stage_complete=False, seconds=monotonic() - started,
            static_inventory=dict(index=encode(self.shard.index), unit=encode(self.shard.unit_header),
                reused_tables=list(_STATIC_TABLES), binding_solver_replayed=True))
        state.check()
        _durable_bytes(out / 'result.json', _json(report), replace=False)
        return report

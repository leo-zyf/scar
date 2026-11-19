"""Read one archived F4 graph shard without expanding the original unit."""

from collections.abc import Iterator
from dataclasses import dataclass
import gzip
from hashlib import sha256
from pathlib import Path

from scar.analysis.program_state_coverage import _graph_contract
from scar.analysis.source_consumer_parents import _stamp
from scar.ir.program import source_productions as P, view as V
from scar.ir.program.dependencies import GraphNamespace
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import _decode, _record_tags, decode, loads


_INDEX = 'scar.source-state-coverage-index'
_UNIT = 'scar.source-state-unit-coverage'


def _regular(path):
    if not path.is_absolute() or not path.is_file() or path.resolve(strict=True) != path:
        raise ValueError('F4 shard receipt requires an original regular file: ' + str(path))
    return path


def _read(artifact, path, schema=None):
    if artifact.uri != str(path) or (schema is not None and artifact.schema != schema):
        raise ValueError('F4 shard receipt has a different original path or schema')
    raw = _regular(path).read_bytes()
    if len(raw) != artifact.size_bytes or sha256(raw).hexdigest() != artifact.sha256:
        raise ValueError('F4 shard receipt does not match the original file: ' + str(path))
    return raw


def _document(raw, schema):
    value = loads(raw)
    if (type(value) is not dict or value.get('schema') != schema
            or type(value.get('schema_version')) is not int or value['schema_version'] != 1):
        raise ValueError('F4 shard requires an original versioned ' + schema + ' document')
    return value


@dataclass(frozen=True, slots=True)
class _TableReceipt:
    path: str
    bytes: int
    sha256: str
    decoded_bytes: int
    decoded_sha256: str
    records: int


def _rows(folder, receipt, cls):
    path = _regular(folder / receipt.path)
    before = _stamp(path)
    tags = _record_tags(P.PRODUCTION_RECORD_TAGS)
    with path.open('rb') as stream:
        compressed = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            compressed.update(block)
        if ((stream.tell(), compressed.hexdigest()) != (receipt.bytes, receipt.sha256)
                or _stamp(path) != before):
            raise ValueError('F4 shard compressed table differs: ' + receipt.path)
        stream.seek(0)
        digest, size, count = sha256(), 0, 0
        with gzip.GzipFile(fileobj=stream) as decoded:
            while (raw := decoded.readline(receipt.decoded_bytes - size + 1)):
                size += len(raw)
                count += 1
                if size > receipt.decoded_bytes or count > receipt.records or not raw.endswith(b'\n'):
                    raise ValueError('F4 shard decoded line or table exceeds its receipt: ' + receipt.path)
                digest.update(raw)
                row = _decode(cls, loads(raw), tags)
                if cls is V.StoredSourceEdge and ((row.region is not None) != row.original or row.ordinal < 0):
                    raise ValueError('F4 shard edge has an invalid original region coordinate')
                yield row
        if ((size, count, digest.hexdigest()) != (receipt.decoded_bytes, receipt.records, receipt.decoded_sha256)
                or _stamp(path) != before):
            raise ValueError('F4 shard decoded table differs: ' + receipt.path)


@dataclass(frozen=True, slots=True)
class StoredSourceGraphShard:
    """Only the saved projection grammar. Original recipes remain separate."""

    unit: str
    namespace: GraphNamespace
    source_material: ProviderArtifact
    catalog: ProviderArtifact
    index: ProviderArtifact
    unit_header: ProviderArtifact
    template_count: int
    relation_count: int
    _folder: Path
    _nodes: _TableReceipt
    _edges: _TableReceipt

    @property
    def node_count(self):
        return self._nodes.records

    @property
    def edge_projection_count(self):
        return self._edges.records

    def iter_nodes(self) -> Iterator[V.StoredSourceNode]:
        """Yield typed nodes. Exhaust the iterator to check decoded receipts."""
        return _rows(self._folder, self._nodes, V.StoredSourceNode)

    def iter_edges(self) -> Iterator[V.StoredSourceEdge]:
        """Yield typed edges. Exhaust the iterator to check decoded receipts."""
        return _rows(self._folder, self._edges, V.StoredSourceEdge)


def open_source_graph_shard(inventory_dir, unit, *, index_sha256) -> StoredSourceGraphShard:
    """Admit the locked headers for one unit, leaving both tables unopened.

    This is typed receipt admission. It does not rederive source semantics,
    original step/region records or the region-local producer indexes.
    """
    if type(unit) is not str or not unit or Path(unit).name != unit or unit in ('.', '..'):
        raise ValueError('F4 shard needs an exact unit identity from the inventory index')
    root = Path(inventory_dir).resolve(strict=True)
    index_path = root / 'index.json'
    raw = _regular(index_path).read_bytes()
    index_artifact = ProviderArtifact(str(index_path), index_sha256, len(raw), _INDEX)
    if sha256(raw).hexdigest() != index_sha256:
        raise ValueError('F4 shard inventory index differs from its declared digest')
    index = _document(raw, _INDEX)
    contract = _graph_contract()
    if (index.get('graph_contract') != contract or index.get('semantic_rederivation') is not False
            or index.get('target_executed') is not False or index.get('first_stage_complete') is not False
            or unit not in index['units']):
        raise ValueError('F4 shard inventory index has a different graph contract or unit domain')
    plan_ref = decode(ProviderArtifact, index['plan'])
    inventory_ref = decode(ProviderArtifact, index['inventory'])
    plan = _document(_read(plan_ref, root / 'plan.json', 'scar.source-state-coverage-plan'), plan_ref.schema)
    inventory = _document(_read(inventory_ref, root / 'result.json', 'scar.source-state-coverage'), inventory_ref.schema)
    selected_units = plan['selected_units']
    if (plan != inventory['plan'] or plan.get('local_graph_shards') is not True
            or plan.get('graph_contract') != contract or set(selected_units) != set(index['units'])
            or len(selected_units) != len(index['units'])):
        raise ValueError('F4 shard inventory plan differs from its published unit domain')
    catalog = decode(ProviderArtifact, plan['parent'])
    _read(catalog, Path(plan['catalog']) / 'result.json', plan['catalog_schema'])

    unit_path = root / 'units' / unit / 'result.json'
    unit_artifact = decode(ProviderArtifact, index['units'][unit])
    header = _document(_read(unit_artifact, unit_path, _UNIT), _UNIT)
    if ((header.get('status'), header.get('unit')) != ('INVENTORIED', unit)
            or type(header.get('original_body_bytes_copied')) is not int or header['original_body_bytes_copied'] != 0
            or any(header.get(field) is not False for field in (
                'target_executed', 'first_stage_complete', 'loops_unrolled', 'binding_fixed_points_complete'))):
        raise ValueError('F4 shard is not a published static graph unit')
    graph = header['graph']
    namespace = decode(GraphNamespace, header['namespace'], record_tags=P.PRODUCTION_RECORD_TAGS)
    material = decode(ProviderArtifact, header['source_material'])
    _read(material, Path(material.uri))
    if (namespace.graph.artifact != material or namespace.source_body is not None
            or namespace.representation != contract['representation'] or graph['contract'] != contract
            or set(graph['tables']) != {'nodes', 'edges'}):
        raise ValueError('F4 shard material, namespace or table contract differs')
    nodes, edges = (decode(_TableReceipt, graph['tables'][name]) for name in ('nodes', 'edges'))
    for name, receipt in (('nodes', nodes), ('edges', edges)):
        if (receipt.path != name + '.jsonl.gz' or receipt.bytes <= 0 or receipt.records < 0
                or receipt.decoded_bytes < receipt.records):
            raise ValueError('F4 shard graph table path or extent differs')
    if ((graph['nodes'], graph['edge_projections'], graph['templates'], graph['relations'])
            != (nodes.records, edges.records, header['counts']['templates'], header['counts']['normalized_relations'])
            or nodes.records != header['counts']['normalized_endpoints']):
        raise ValueError('F4 shard graph tables differ from their typed record counts')
    return StoredSourceGraphShard(unit, namespace, material, catalog, index_artifact, unit_artifact,
                                  graph['templates'], graph['relations'], unit_path.parent / 'graph', nodes, edges)

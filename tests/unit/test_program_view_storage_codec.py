"""Bulk decoding retains exact fields/IDs with one local tag admission."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scar.analysis import program_view_storage as storage
from scar.ir.program import view as V, source_productions as P
from scar.ir.program.dependencies import DependencyKind as K, DependencyMeaning as M, DependencyStrength as S
from scar.ir.program.ids import RegionID, OperationID
from scar.ir.program.source_declarations import SourceTemplateKey
from scar.ir.record_codec import encode


def test_bulk_read_admits_tags_once_without_weakening_row_validation(monkeypatch):
    key = SourceTemplateKey(0, P.ProductionPurpose.ROOT)
    root, operation = RegionID('root'), OperationID('operation')
    node = V.StoredSourceNode(root, key, P.ProductionRegionID('original'),
        V.ProgramEntityKind.REGION, V.ProgramNodeRole.REGION, None, None)
    edge = V.StoredSourceEdge(root, operation, K.CONTAINS, M.STRUCTURE, S.EXACT,
        key, 0, 'test.original-owner', False, (), None)
    records = dict(nodes=(node,) * 5, edges=(edge,) * 5)
    rows = {name: [encode(record, record_tags=P.PRODUCTION_RECORD_TAGS) for record in values]
        for name, values in records.items()}
    monkeypatch.setattr(storage, 'pack_nodes', lambda provider: iter(records['nodes']))
    monkeypatch.setattr(storage, 'pack_edges', lambda provider: iter(records['edges']))
    monkeypatch.setattr(storage, '_table', lambda folder, name: iter(rows[name]))
    admitted = []
    original_tags = storage._record_tags

    def tags(value):
        admitted.append(value)
        return original_tags(value)

    monkeypatch.setattr(storage, '_record_tags', tags)
    provider = SimpleNamespace(check=lambda: None)
    assert storage.read_and_validate(None, dict(nodes='nodes', edges='edges'), provider) == dict(nodes=5, edges=5)
    assert len(admitted) == 1
    original = deepcopy(rows)
    rows['nodes'][0]['extra'] = 'not part of the wire contract'
    with pytest.raises(ValueError):
        storage.read_and_validate(None, dict(nodes='nodes', edges='edges'), provider)
    rows = deepcopy(original)
    rows['edges'][0]['source']['wire'] = 'program_operation:root'
    with pytest.raises(ValueError):
        storage.read_and_validate(None, dict(nodes='nodes', edges='edges'), provider)
    assert len(admitted) == 3

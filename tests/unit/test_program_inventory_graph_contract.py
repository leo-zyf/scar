"""An unused solver exception never authorizes changed graph wire rules."""
from copy import deepcopy
import json

import pytest

from scar.analysis import program_inventory_admission as admission
from scar.analysis import program_state_coverage as coverage


def test_explicit_cache_root_admits_its_own_report_and_children_once(tmp_path):
    from scar.analysis.build_f4 import _reports
    (tmp_path/'result.json').write_text(json.dumps(dict(schema='test.parent')))
    child=tmp_path/'child';child.mkdir()
    (child/'result.json').write_text(json.dumps(dict(schema='test.child')))
    rows=_reports((tmp_path,child,tmp_path))
    assert {(path,report['schema'])for path,report in rows}=={
        (tmp_path,'test.parent'),(child,'test.child')}
    assert len(rows)==2


def test_inventory_graph_contract_is_checked_separately_from_unused_solver(tmp_path,monkeypatch):
    contract=coverage._graph_contract()
    pins=dict(graph_mapping=dict(types='current-graph-types'),wire_codec='current-codec',
        table_output='current-table-writer',state=dict(definitions='current-unused-solver',state='state'))
    monkeypatch.setattr(admission,'_pins',lambda:deepcopy(pins))
    plan=dict(schema='scar.source-state-coverage-plan',schema_version=1,
        pins=deepcopy(pins),binding_fixed_points_run=False,local_graph_shards=True,
        graph_contract=deepcopy(contract),selected_units=[])
    assert admission.compatible_static_pins(plan)
    old_solver=deepcopy(plan)
    old_solver['pins']['state']['definitions']='previous-unused-solver'
    assert admission.compatible_static_pins(old_solver)
    changed=deepcopy(old_solver)
    changed['pins']['graph_mapping']['types']='different-graph-types'
    assert not admission.compatible_static_pins(changed)
    for name in ('wire_codec','table_output'):
        changed=deepcopy(old_solver);changed['pins'][name]='different-used-rule'
        assert not admission.compatible_static_pins(changed)
    changed=deepcopy(old_solver)
    changed['graph_contract']['mapping']['original_edge_coordinates']=['template','flat_ordinal']
    assert not admission.compatible_static_pins(changed)
    # Reindexing cannot turn the changed grammar into a current receipt,
    # even before there are any local graph tables to read.
    report=dict(schema='scar.source-state-coverage',schema_version=1,plan=changed)
    (tmp_path/'plan.json').write_text(json.dumps(changed))
    (tmp_path/'result.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='graph wire/mapping contract'):
        coverage._inventory_index(tmp_path)

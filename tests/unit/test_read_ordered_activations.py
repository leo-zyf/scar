"""A rewritten valid receipt cannot replace original typed semantics."""
from dataclasses import replace

import pytest

from scar.analysis import ordered_loop_transfer_storage as reader
from scar.artifacts import write_table
from scar.ir.program.ordered_regions import OrderedInputBoundary


def test_independent_table_comparison_rejects_rehashed_semantic_change(tmp_path,monkeypatch):
    original=OrderedInputBoundary('#/inputs/0','ORIGINAL_SOURCE_BOUNDARY')
    monkeypatch.setattr(reader,'_activation_rows',lambda _:{
        'input_boundaries':(OrderedInputBoundary,(original,))})
    path=tmp_path/'input_boundaries.jsonl.gz'
    report={'tables':{'input_boundaries':write_table(path,(original,))}}
    assert reader._activation_tables(tmp_path,report,object())=={'input_boundaries':1}
    path.unlink()
    report['tables']['input_boundaries']=write_table(path,(replace(original,reason='FORGED_BOUNDARY'),))
    with pytest.raises(ValueError,match='differs from its original source derivation'):
        reader._activation_tables(tmp_path,report,object())

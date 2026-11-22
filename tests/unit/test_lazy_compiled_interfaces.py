"""Targeted phase/guard checks using a retained leaf, with no source rebuild."""
import gzip
from hashlib import sha256
import json
from pathlib import Path
import shutil

import pytest

from scar.analysis.lazy_compiled_interfaces import LazyCompiledInterfaces
from scar.analysis.observed_code_correspondence import ObservedCompiledLocalMaterial


@pytest.fixture(scope='module')
def retained_leaf():
    root=Path(__file__).resolve().parents[2]/'artifacts/current/experiments/cpu-gray-scott-current/graph-frontend-current'
    catalog=root/'complete-source-catalog/result.json'
    if not catalog.is_file():pytest.skip('existing receipt-bound external leaf required; test never rebuilds a frontend')
    report=json.loads(catalog.read_bytes())
    entries=[(json.loads(Path(row['material']['path']).read_bytes())['tables']['facts']['decoded_bytes'],row)
        for row in report['compiled']]
    _,entry=min(entries,key=lambda item:item[0])
    material=ObservedCompiledLocalMaterial(Path(entry['material']['path']).parent)
    return material,Path(entry['interfaces']['path']).parent


def _copy_interface_header(source,out):
    out.mkdir();report=json.loads((source/'result.json').read_bytes())
    shutil.copyfile(source/'result.json',out/'result.json')
    shutil.copyfile(source/report['row']['interfaces'],out/report['row']['interfaces'])
    return report


def test_cold_receipts_do_not_decode_facts_and_hot_reader_is_reused(retained_leaf,tmp_path,monkeypatch):
    material,original=retained_leaf;folder=tmp_path/'interface'
    report=_copy_interface_header(original,folder);calls=[]
    original_load=material.fact_store.load
    def observed_load(*args,**kwargs):
        calls.append(args);return original_load(*args,**kwargs)
    monkeypatch.setattr(material.fact_store,'load',observed_load)
    reader=LazyCompiledInterfaces(folder,material)
    reader.validate_parents()
    assert not calls and not reader.is_loaded and not reader.loaded
    assert reader.stats==dict(unit_reads=0,iteration_reads=0,body_reads=0,mapping_reads=0)
    assert dict(reader.declared_counts)==report['counts']
    assert reader.admission_summary()['status']=='HEADER_RECEIPTS_ADMITTED_TYPES_DEFERRED'
    # A typed unit is supplied by the same original source material; the lazy
    # boundary still must construct the unchanged strict interface reader.
    unit=material.fact_store.load(material.unit)
    value=reader.load(unit)
    assert calls and reader.is_loaded and value.unit==material.unit
    assert reader.stats['unit_reads']==1
    assert reader.admission_summary()['status']=='ORIGINAL_READER_TYPES_AND_WITNESSES_VALIDATED'
    assert value is reader.load(unit) and reader.stats['unit_reads']==1
    # Input changes are refused both after hot loading and before first use.
    cold_folder=tmp_path/'cold-interface';_copy_interface_header(original,cold_folder)
    cold=LazyCompiledInterfaces(cold_folder,material)
    table=cold_folder/report['row']['interfaces'];table.write_bytes(table.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='input changed'):cold.validate_parents()
    assert not cold.is_loaded and not cold.loaded
    hot_table=folder/report['row']['interfaces'];hot_table.write_bytes(hot_table.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='input changed'):reader.load(unit)


def test_failed_original_type_validation_never_publishes_a_loaded_reader(retained_leaf,tmp_path):
    material,original=retained_leaf;folder=tmp_path/'bad-typed-interface'
    report=_copy_interface_header(original,folder);table=folder/report['row']['interfaces']
    document=json.loads(gzip.decompress(table.read_bytes()));document['unit']='not-the-original-unit'
    decoded=json.dumps(document,sort_keys=True,separators=(',',':')).encode();payload=gzip.compress(decoded,mtime=0)
    table.write_bytes(payload)
    report['row'].update(bytes=len(payload),sha256=sha256(payload).hexdigest(),
        decoded_bytes=len(decoded),decoded_sha256=sha256(decoded).hexdigest())
    (folder/'result.json').write_text(json.dumps(report,sort_keys=True,separators=(',',':'))+'\n')
    reader=LazyCompiledInterfaces(folder,material)
    assert not reader.is_loaded
    unit=material.fact_store.load(material.unit)
    with pytest.raises(ValueError,match='unit identity differs'):reader.load(unit)
    assert not reader.is_loaded and not reader.loaded and reader.stats['unit_reads']==0

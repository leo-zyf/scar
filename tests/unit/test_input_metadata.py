"""Observed call metadata supplies shapes without storing array contents."""
from hashlib import sha256
import json
import numpy as np
import pytest

from scar.compiler.input_metadata import InputMetadataObserver,shapes_from_metadata
from scar.compiler.model import SourceRef

def test_bounded_metadata_and_source_checked_shape_selection(tmp_path):
    source=SourceRef(str(tmp_path/'program.py'),'1'*64,'any_name',1,0,'original-code')
    value=np.arange(36,dtype=np.float64).reshape(6,6)
    observer=InputMetadataObserver(max_calls_per_code=2)
    assert observer(source,('x',),(value,)) is True
    value[:]=100
    assert observer(source,('x',),(value,)) is False
    assert observer(source,('x',),(value,)) is False
    path=tmp_path/'metadata.json';observer.write(path)
    document=json.loads(path.read_text())
    assert len(document['records'])==2 and not document['tensor_contents_recorded']
    assert set(document['records'][0]['inputs']['x'])=={
        'kind','shape','dtype','strides','nbytes','c_contiguous','aligned','device','contents_recorded'}
    shapes,evidence=shapes_from_metadata(path,source,('x',))
    assert shapes=={'x':[6,6]} and evidence['sha256']==sha256(path.read_bytes()).hexdigest()
    assert evidence['all_future_calls_proven'] is False
    document['records'][1]['inputs']['x']['shape']=[3,12]
    document['records'][1]['inputs']['x']['strides']=[96,8]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match='polymorphic'):
        shapes_from_metadata(path,source,('x',))

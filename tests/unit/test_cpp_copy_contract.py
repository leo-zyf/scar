"""One native ABI test for the exact ndarray-copy/fresh-output boundary."""
from hashlib import sha256
import importlib.util

import numpy as np
import pytest

from scar.codegen.cpp import build
from scar.compiler.array_copy_contract import copy_attributes,COPY_REQUIRED_GUARDS
from scar.compiler.model import ArrayRegion,Expr,ExecutionPlan,SourceRef
from scar.compiler.runtime import guarded_function
from scar.compiler.validation import InvalidPlan,validate_region
from scar.trace.code_snapshot import code_key


def test_copy_native_fresh_recomputed_and_custom_method_fallback(tmp_path):
    path=tmp_path/'original_copy.py';path.write_text('def copy_value(value):\n    return value.copy()\n')
    spec=importlib.util.spec_from_file_location('original_copy',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=module.copy_value
    source=SourceRef(str(path),sha256(path.read_bytes()).hexdigest(),original.__qualname__,1,0,code_key(original.__code__))
    region=ArrayRegion('copy-region',source,('value',),(
        Expr('input','input',(),{'name':'value'}),Expr('copy','copy',('input',),copy_attributes())),
        'copy',assumptions=ArrayRegion.__dataclass_fields__['assumptions'].default+COPY_REQUIRED_GUARDS)
    validate_region(region)
    artifact=build(ExecutionPlan(region,cost={'memory_budget_bytes':1<<20}),tmp_path/'native')
    wrapped=guarded_function(original,artifact,exclusive_inputs=True,trial=True)
    value=np.arange(30,dtype=np.float64).reshape(5,6)
    first=wrapped(value);assert first.tobytes()==original(value).tobytes()
    assert first.flags.c_contiguous and first.flags.owndata and not np.shares_memory(first,value)
    first[:]=-17;second=wrapped(value)
    assert second.tobytes()==original(value).tobytes()and not np.shares_memory(first,second)
    value[:]+=3;third=wrapped(value)
    assert third.tobytes()==original(value).tobytes()and not np.shares_memory(third,value)
    assert wrapped.scar_stats['native_calls']==3

    class CustomCopy:
        def __init__(self,array):self.array=array;self.calls=0
        def copy(self):self.calls+=1;return self.array.copy()
    custom=CustomCopy(value);actual=wrapped(custom)
    assert actual.tobytes()==value.tobytes()and custom.calls==1
    # Module aliases are mutable. Replacing one must not authorize this custom
    # class as a native ndarray or suppress its side effect.
    anchor=np.ndarray;np.ndarray=CustomCopy
    try:
        actual=wrapped(custom)
        assert actual.tobytes()==value.tobytes()and custom.calls==2
    finally:np.ndarray=anchor
    assert wrapped.scar_stats['native_calls']==3 and wrapped.scar_stats['fallback_calls']==2

    from dataclasses import replace
    with pytest.raises(InvalidPlan):validate_region(replace(region,assumptions=region.assumptions[:-1]))
    with pytest.raises(InvalidPlan):validate_region(replace(region,nodes=(region.nodes[0],
        Expr('copy','copy',('input',),{**copy_attributes(),'order':'K'}))))

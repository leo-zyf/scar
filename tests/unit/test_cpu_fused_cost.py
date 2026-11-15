"""Canonical feature identities preserve alias geometry without source IDs."""
from copy import deepcopy

import pytest

from scar.codegen.cpu_fused_cost import _canonical_cases, _layouts, _regimes, fused_features


def test_canonical_fused_features_keep_dependency_and_backing_geometry_explicit():
    cases=_canonical_cases()
    assert len(cases)==52 and len([row for row in cases if row['holdout']])==3
    case=next(row for row in cases if row['array_loads']==4 and row['family']=='INDEPENDENT_CHAINS')
    parameters=dict(rows=32,cols=31,workers=4,arithmetic=case['arithmetic'],array_loads=4,
        output_stores=1,dependent_operation_depth=case['dependent_operation_depth'],
        operand_layouts=_layouts(4,32,31,'INDEPENDENT'),backing_storage_groups=4)
    independent,independent_geometry=fused_features(parameters,2**20)
    shared=deepcopy(parameters);shared['operand_layouts']=_layouts(4,32,31,'SHARED_SHIFTED');shared['backing_storage_groups']=1
    features,geometry=fused_features(shared,2**20)
    assert features['logical_loads']==independent['logical_loads']==32*31
    assert features['dependency_depth']==32*31/4*case['dependent_operation_depth']
    assert geometry['backing_bytes']<independent_geometry['backing_bytes']
    assert features['row_padding']>0 and not geometry['logical_bytes_are_dram_bytes']
    renamed=deepcopy(shared)
    for layout in renamed['operand_layouts']:layout['storage_group']=7
    assert fused_features(renamed,2**20)==(features,geometry)
    bad=deepcopy(parameters);bad['dag_sha256']='f'*64
    with pytest.raises(ValueError,match='generic fused geometry'):fused_features(bad,2**20)
    bad=deepcopy(parameters);bad['operand_layouts'][0].pop('storage_group')
    with pytest.raises(ValueError,match='backing groups'):fused_features(bad,2**20)
    bad=deepcopy(parameters);bad['arithmetic']['neg']=1
    with pytest.raises(ValueError,match='uncalibrated'):fused_features(bad,2**20)
    # A replayable fit still cannot pass precision admission on poor holdouts.
    observations=[]
    for groups in (1,2):
        p=deepcopy(parameters);p['array_loads']=groups;p['backing_storage_groups']=groups
        p['operand_layouts']=_layouts(groups,32,31,'INDEPENDENT')
        values,props=fused_features(p,2**20)
        for heldout in (False,True):
            observations.append(dict(primitive='declared-unit-policy-check',geometry='INDEPENDENT',
                holdout=heldout,parameters=p,features=values,geometry_properties=props,
                median_ns=1000 if heldout else 100))
    models=_regimes(observations,2**20,0.20)
    assert all(not model['admitted'] and model['maximum_holdout_relative_error']>0.20
               for model in models.values())

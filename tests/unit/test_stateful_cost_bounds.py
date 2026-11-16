import pytest

from scar.optimization.stateful_cost_bounds import _relaxation


def test_nonnegative_relaxation_is_partial_and_strict_dominance_keeps_ties():
    terms=[dict(id='a',key='A',component='allocation',count=2),
        dict(id='u',key='U',component='unknown_fused',count=1),
        dict(id='b',key='B',component='guard',count=1)]
    known={'A':dict(median_ns=10.25,basis='DIRECT_COMPONENT_TIMING',fact={'source':'a'}),
        'B':dict(median_ns=4.5,basis='DECLARED_COMPONENT_MODEL',fact={'source':'b'})}
    result=_relaxation(terms,known,24)
    assert result['ordered_known_component_sum_ns']==25
    assert result['interval_ticks']==dict(lower=25,upper=None,upper_kind='POSITIVE_INFINITY')
    assert result['unknown_terms']==['u'] and result['strictly_dominated'] is True
    assert result['components'][1]['authority']=='NONNEGATIVITY_RELAXATION_NOT_A_DURATION_ESTIMATE'
    assert result['components'][1]['fact'] is None and result['duration_profile_emitted'] is False
    assert _relaxation(terms,known,25)['strictly_dominated'] is False
    # All nonnegative completions respect the same lower endpoint.
    for remaining in (0,.25,1.,10000.):
        complete=dict(known,U=dict(median_ns=remaining,basis='DECLARED_COMPONENT_MODEL',fact={'source':'u'}))
        assert _relaxation(terms,complete,25)['interval_ticks']['lower']>=25
    with pytest.raises(ValueError,match='nonnegative'):
        _relaxation(terms,{**known,'A':dict(median_ns=-1,basis='bad',fact={})},24)

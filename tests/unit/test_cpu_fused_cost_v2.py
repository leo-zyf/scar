from scar.codegen.cpu_fused_cost_models import (
    _V2_POLICY, _configurations_v2, _configurations_v3, _predict_v3, _source_v2,
    dag_features, registry_v2, registry_v3,
)


def test_frozen_anonymous_dag_design_preserves_dependencies_and_shared_roots():
    registry=registry_v2()
    training=[r for r in registry if not r['holdout']]
    holdout=[r for r in registry if r['holdout']]
    assert len(training)==64 and len(holdout)==8
    assert not {r['id'] for r in training}&{r['id'] for r in holdout}
    assert _V2_POLICY['maximum_holdout_relative_error']==.20
    context={'worker_cpu_ids':[0,1,2,3]}
    configs=_configurations_v2(context,64<<20)
    assert len(configs)==416
    assert not {(r['parameters']['rows'],r['parameters']['cols']) for r in configs if r['holdout']}&{
        (r['parameters']['rows'],r['parameters']['cols']) for r in configs if not r['holdout']}
    chain=next(r for r in training if r['family']=='SERIAL_FANIN' and r['array_loads']==8 and r['scalar_count']==0)
    balanced=next(r for r in training if r['family']=='BALANCED_FANIN' and r['array_loads']==8 and r['scalar_count']==0)
    layouts=[{'stride':515,'offset':i,'storage_group':0} for i in range(8)]
    a,p,_=dag_features(chain['graph'],chain['output'],layouts,513,513,4,64<<20)
    b,q,_=dag_features(balanced['graph'],balanced['output'],layouts,513,513,4,64<<20)
    assert p['arithmetic']==q['arithmetic'] and p['array_loads']==q['array_loads']
    assert a['path_addsub']>b['path_addsub'] and a['width']<b['width']
    # One compiler-visible shared root supplies every shifted load in this case.
    source=_source_v2(context)
    start=source.index('void '+chain['id']+'_g1(')
    fragment=source[start:source.index('\n}',start)]
    assert 'a.inputs[0]' in fragment and 'a.inputs[1]' not in fragment


def test_bottleneck_composition_uses_max_and_independent_fresh_design():
    registry=registry_v3();configs=_configurations_v3({'worker_cpu_ids':[0,1,2,3]})
    assert len([r for r in configs if r['role']=='BARRIER'])==1
    assert len([r for r in configs if r['holdout']])==32
    assert {(r['rows']%2,r['cols']%2) for r in configs if r['holdout']}=={(0,0),(1,1)}
    observed_v2={r['id'] for r in registry_v2() if r['holdout']}
    assert not observed_v2&{r['id'] for r in registry if r['holdout']}
    case={'graph':[{'id':'v0','op':'add','inputs':['x0','s0']},
                   {'id':'v1','op':'mul','inputs':['v0','s1']}],'output':'v1'}
    # Synthetic unit values exercise only composition, never performance claims.
    coefficients={'barrier_ns':7,'memory':{'1':{'1':[{'elements':1,'proxy_ns_per_element':4},
        {'elements':100,'proxy_ns_per_element':4}]}},'arithmetic_ns_per_operation':{'add':2,'sub':2,'mul':3},
        'dependency_ns_per_operation':{'add':1,'sub':1,'mul':2}}
    result=_predict_v3(case,{'groups':1,'elements':10,'array_loads':1,
        'arithmetic':{'add':1,'sub':0,'mul':1}},coefficients)
    assert result['duration_ns']==7+max(40,50,30)
    assert result['dominant_proxy']=='ARITHMETIC'

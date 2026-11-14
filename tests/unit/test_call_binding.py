"""A bounded exact site observes targets, without profiling their bodies."""
from copy import deepcopy
import dis,json,sys

import pytest

from scar.compiler.call_binding import write_call_binding_plan,join_call_bindings
from scar.trace.call_binding import CallBindingCollector,target_metadata
from scar.trace.code_snapshot import code_key,site
from scar.trace.lean_capture import LeanCapturePlan,LeanBodyPlan


def test_local_call_budget_original_offset_and_unknown_native_fields(tmp_path,monkeypatch):
    if sys.implementation.name!='cpython' or sys.version_info[:3]!=(3,14,7):
        pytest.skip('pinned monitoring CALL contract required')
    path=tmp_path/'original.py'
    text=('def leaf(x):\n    return x+1\n\n'
          'def selected():\n    for i in range(5):\n        leaf(i)\n\n'
          'def unselected():\n    return leaf(8)\n')
    path.write_text(text);root=compile(text,str(path),'exec',dont_inherit=True)
    caller=next(c for c in root.co_consts if type(c)is type(root) and c.co_name=='selected')
    original=LeanCapturePlan.from_codes(tmp_path,(caller,));body=original.bodies[0]
    offset=[i.offset for i in dis.get_instructions(caller) if i.opname=='CALL'][-1]
    selector=LeanCapturePlan(str(tmp_path),(LeanBodyPlan(body.source,body.code,(offset,),()),))
    plan=dict(schema='scar.call-binding-plan',schema_version=1,max_targets_per_site=2,selector=selector.to_dict(),catalog={},frontiers=[],
        sites=[dict(id='unit:call:0',unit='unit',query=0,code_key=body.code.key,offset=offset,site=site(caller,offset),
            original_step='original-call',original_template={'occurrence':4,'purpose':'VALUE'},result_port=9,fact_reference={})])
    plan_path=tmp_path/'plan.json';write_call_binding_plan(plan,plan_path)
    def forbidden(*args,**kwargs):raise AssertionError('global call hook is forbidden')
    monkeypatch.setattr(sys,'setprofile',forbidden);monkeypatch.setattr(sys.monitoring,'set_events',forbidden)
    collector=CallBindingCollector(plan_path,tmp_path/'capture').start()
    try:
        namespace={};exec(compile(text,str(path),'exec',dont_inherit=True),namespace)
        namespace['selected']();namespace['unselected']()
        assert collector.status=='COMPLETE' and collector.counts=={'unit:call:0':2}
        assert sys.monitoring.get_events(collector.tool_id)==0
        assert all(sys.monitoring.get_local_events(collector.tool_id,c)==0 for c in collector.codes.values())
        assert [r['offset'] for r in collector.records]==[offset,offset]
        assert all(r['target']['code']['key']==code_key(namespace['leaf'].__code__) for r in collector.records)
        joined=join_call_bindings(plan,collector.document())
        assert joined['observed_sites']==1 and joined['attempts']==2
        assert all(r['original_call']['step']=='original-call' for r in joined['bindings'])
        assert not joined['first_stage_complete']
        changed=deepcopy(collector.document());changed['records'][0]['offset']+=2
        with pytest.raises(ValueError,match='code/offset'):join_call_bindings(plan,changed)
    finally:collector.close()
    saved=json.loads(next((tmp_path/'capture').glob('call-bindings.*.json')).read_text())
    assert saved['collector']['closed'] and not saved['collector']['array_values_collected']
    class Hostile:
        def __getattribute__(self,name):raise AssertionError('application getter')
        def __repr__(self):raise AssertionError('application repr')
        def __call__(self):return None
    target=target_metadata(Hostile())
    assert target['code']is None and target['native_boundary']['status']=='NATIVE_SCHEMA_AND_IMPLEMENTATION_UNRESOLVED'
    assert target_metadata(len)['native_boundary']['status']=='EXACT_CARRIER_OBSERVED_SCHEMA_NOT_JOINED'
    # Matching filename/qualname is insufficient when the full CodeID differs.
    mismatch=deepcopy(plan);wrong='pycode-v2:'+'0'*64
    mismatch['selector']['bodies'][0]['code']['key']=wrong
    mismatch['sites'][0]['code_key']=wrong;mismatch['sites'][0]['site']['code']=wrong
    mismatch_path=tmp_path/'mismatch.json';write_call_binding_plan(mismatch,mismatch_path)
    rejected=CallBindingCollector(mismatch_path,tmp_path/'mismatch-capture').start()
    try:
        namespace={};exec(compile(text,str(path),'exec',dont_inherit=True),namespace);namespace['selected']()
        assert not rejected.records and any(r['error']=='CODE_ID_MISMATCH' for r in rejected.errors)
    finally:rejected.close()
    # Optional metadata must never disappear on the unsupported legacy route.
    from scar.analysis.build import build
    with pytest.raises(ValueError,match='legacy backend'):
        build(path,None,tmp_path/'legacy-build',model=tmp_path/'unused-model',
            bounded_call_bindings=(tmp_path/'capture',))
    assert not (tmp_path/'legacy-build').exists()

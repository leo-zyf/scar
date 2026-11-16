import json

import pytest

from scar.ir.execution_recipe import artifact_reference
from scar.ir.program.providers import ProviderFactReference
from scar.ir.record_codec import decode
from scar.optimization.interval_choices import CostInterval,IntervalOption,solve_interval_choices


def test_sourced_intervals_strict_exclusion_crossing_frontier_and_ties(tmp_path):
    contract='e'*64;scope={'call_scope':'ONE_CALL','same_source_contract':True}
    legality=dict(contract_sha256=contract,resource_demands={'cpu':1},conditions=['ORIGINAL_GUARDS'])
    counter=0
    def solve(intervals):
        nonlocal counter
        packets={name:dict(interval_ticks=dict(lower=low,upper=high),tick_seconds=1e-9)
            for name,(low,high)in intervals.items()}
        path=tmp_path/(str(counter)+'.json');counter+=1
        path.write_text(json.dumps(dict(schema='test.interval-facts',scope=scope,legality=legality,packets=packets)))
        artifact=artifact_reference(path,'test.interval-facts')
        def ref(locator):return decode(ProviderFactReference,dict(artifact=artifact,locator=locator))
        options=[IntervalOption(name,contract,scope,CostInterval(low,high),{'cpu':1},('ORIGINAL_GUARDS',),
            (ref('#/packets/'+name),),(ref('#/scope'),),(ref('#/legality'),))for name,(low,high)in intervals.items()]
        return solve_interval_choices(options,contract_sha256=contract,scope=scope,tick_seconds=1e-9),options
    result,_=solve({'known':(5,5),'strict':(6,None)})
    assert result['status']=='FINITE_MODEL_OPTIMAL' and result['selected_id']=='known'
    assert result['strictly_excluded_option_ids']==['strict'] and result['retained_option_count']==2
    result,_=solve({'known':(5,5),'crossing':(4,None),'strict':(6,None)})
    assert result['status']=='UNRESOLVED' and result['selected_id']is None
    assert [r['option_id']for r in result['remaining_frontier']]==['crossing']
    result,_=solve({'known':(5,5),'possible_tie':(5,None)})
    assert result['status']=='FINITE_MODEL_OPTIMAL' and result['strictly_excluded_option_ids']==[]
    assert result['possible_equal_cost_option_ids']==['possible_tie']
    result,options=solve({'b':(5,5),'a':(5,5)})
    assert result['selected_id']=='a' and result['unique_optimum_proven']is False
    assert result['deployment_authorized']is False and result['unknown_duration_imputed']is False
    bad=IntervalOption('b',contract,scope,CostInterval(1,1),{'cpu':1},('ORIGINAL_GUARDS',),
        options[0].interval_evidence,options[0].scope_evidence,options[0].legality_evidence)
    with pytest.raises(ValueError,match='matching retained cost interval'):
        solve_interval_choices([bad],contract_sha256=contract,scope=scope,tick_seconds=1e-9)

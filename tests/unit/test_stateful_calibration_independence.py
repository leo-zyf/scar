"""Only explicitly independent resource facts can fill stateful CPU costs."""
import json

from scar.optimization import stateful_cpu_model as M


def test_target_trials_or_missing_independence_remain_rejected_unknown_costs(tmp_path,monkeypatch):
    parameters=dict(rows=4,cols=8,workers=1)
    term=dict(id='term:0',component='ROW_COPY_STAGE',parameters=parameters,
        key=M.component_key('ROW_COPY_STAGE',parameters),count=1,source_ref=None)
    demand=dict(context={'call_scope':'ONE_ORIGINAL_METHOD_CALL_SITE'},
        resource_context={'worker_cpu_ids':[0]},terms=[term])
    monkeypatch.setattr(M,'stateful_cpu_demand',lambda *_:demand)
    for index,flags in enumerate((dict(target_workload_executed=False,target_variants_ranked=False),
            dict(target_workload_executed=True,target_variants_ranked=False),
            dict(target_workload_executed=False,target_variants_ranked=True),{})):
        calibration=dict(schema='scar.stateful-resource-calibration',schema_version=1,
            resource_context=demand['resource_context'],components=[dict(component=term['component'],
                parameters=parameters,basis='DIRECT_COMPONENT_TIMING',median_ns=123)],**flags)
        path=tmp_path/('calibration-'+str(index)+'.json');path.write_text(json.dumps(calibration))
        report=M._predict_stateful_cpu_model({},'candidate',[path],out=tmp_path/('model-'+str(index)))
        if index==0:
            assert report['complete'] and report['duration_ticks']==123 and report['profile']is not None
            assert report['components'][0]['measurement']['duration_ns']==123
            assert not report['rejected_calibrations']
        else:
            assert report['complete']is False and report['duration_ticks']is None and report['profile']is None
            assert report['components'][0]['status']=='MODEL_REJECTED'
            assert report['components'][0]['measurement']is None
            assert report['unresolved'][0]['rejected_estimates'][0]['point_estimate_ns']==123
            assert 'requires explicit False'in report['rejected_calibrations'][0]['reason']

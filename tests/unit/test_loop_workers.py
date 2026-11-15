"""Synthetic sourced models exercise worker choice; no workload is executed."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from test_cpu_hardware_placement import hardware
from test_loop_layout import setup_layout
from scar.codegen.loop import loop_implementation_sha256
from scar.optimization import bind_cpu_placement, propose_cpu_placements
from scar.optimization.loop_layout import plan_loop_layout
from scar.optimization.loop_strategy import loop_execution_plan
from scar.optimization.loop_workers import (BOUNDARY, CALIBRATION_SCOPE, features_for_loop,
    loop_worker_feature_units, plan_loop_workers, selected_loop_worker, verify_loop_worker_choice)


def test_sourced_temporal_family_selects_a_typed_binding_and_keeps_missing_cost_frontiers(hardware,tmp_path):
    loop,strategy,bound,primitive=setup_layout(hardware,tmp_path)
    # Retained single-CPU test declarations can describe layout primitives for
    # either OpenMP team. These are not recorded physical timing observations.
    source=tmp_path/'layout-facts-9.json'; facts=json.loads(source.read_bytes())
    facts['context']['cpu_ids']=[0]; primitive['context']['cpu_ids']=[0]
    source.write_text(json.dumps(facts)); raw=source.read_bytes()
    def update(value):
        if isinstance(value,dict):
            if 'artifact' in value:
                value['artifact'].update(sha256=sha256(raw).hexdigest(),size_bytes=len(raw))
            for item in value.values(): update(item)
        elif isinstance(value,list):
            for item in value: update(item)
    update(primitive)
    base=loop_execution_plan(loop,strategy)
    binding=next(item for item in propose_cpu_placements(base,hardware,max_threads=4,memory_budget_bytes=144)
                 if len(item.cpu_ids)==4)
    four=bind_cpu_placement(base,binding)
    plans=[bound,four]; layouts=[plan_loop_layout(loop,strategy,plan,hardware_profile=primitive) for plan in plans]
    work=features_for_loop(loop,strategy,bound,layouts[0])
    assert work['values']['packed_arithmetic.neg']==240 and work['values']['scalar_arithmetic.neg']==60
    assert work['values']['packed_finite_checks']==32 and work['values']['scalar_finite_checks']==8
    assert work['values']['logical_input_bytes']==144 and not work['worker_scaling_inferred']
    units=loop_worker_feature_units(); names=['packed_iteration_groups']
    profiles=[]
    for index,(plan,layout) in enumerate(zip(plans,layouts)):
        binding=plan.resource_binding
        context=dict(target_path=binding.target_path,target_sha256=binding.target_sha256,executor_id=binding.executor_id,
            cpu_ids=list(binding.cpu_ids),loop_implementation_sha256=loop_implementation_sha256(),
            compiler=primitive['context']['compiler'],layout={key:layout['selected'][key] for key in ('isa','simd_lanes','vector_chains')},
            input_layout='C_CONTIGUOUS_FLOAT64_2D',execution_boundary=BOUNDARY,memory_policy='UNBOUND',
            working_set_regime='SYNTHETIC_UNIT_TEST_MODEL_NOT_HARDWARE_MEASUREMENT')
        axes=dict(rows=2,columns=9,steps=3,body_dependency_depth=work['body_dependency_depth'],
            **{'arithmetic.'+op:value for op,value in work['arithmetic_per_element'].items()})
        domain={key:[value,value] for key,value in axes.items()}
        calibration=dict(scope=CALIBRATION_SCOPE,target_program_executed=False,feature_names=['intercept',*names],
            coefficient_units=dict(intercept='ticks',**{name:units[name] for name in names}),design_rows=[[1,6]],
            included_boundary={name:True for name in ('real_iteration_loop','packed_and_scalar_finite_guards',
                'affinity_and_fenv','team_launch_and_join','fresh_output_allocation')},
            validation=dict(rank=1,parameters=2,parameters_unique=False,basis='SYNTHETIC_MODEL_FIXTURE'))
        models=dict(tick_seconds=dict(value=1e-9,unit='seconds/tick'),intercept=dict(value=10,unit='ticks'),
                    coefficient=dict(value=8 if index==0 else 3,unit=units[names[0]]))
        record=dict(schema='scar.test-loop-worker-fit',context=context,applicability=domain,calibration=calibration,parameters=models)
        path=tmp_path/f'fit-{index}.json'; path.write_text(json.dumps(record)); payload=path.read_bytes()
        artifact=dict(uri=str(path),sha256=sha256(payload).hexdigest(),size_bytes=len(payload),schema=record['schema'])
        def reference(pointer): return dict(artifact=artifact,locator=pointer)
        def parameter(name): return dict(**models[name],evidence=[reference('#/parameters/'+name)])
        profiles.append(dict(schema='scar.loop-worker-cost-profile',schema_version=1,context=context,
            context_evidence=[reference('#/context')],applicability=domain,applicability_evidence=[reference('#/applicability')],
            calibration=calibration,calibration_evidence=[reference('#/calibration')],model=dict(
                kind='EFFECTIVE_AFFINE_TEMPORAL_CALL',feature_names=names,tick_seconds=parameter('tick_seconds'),
                intercept=parameter('intercept'),coefficients={names[0]:parameter('coefficient')})))
    variants=[dict(id=f'cpu-{plan.cpu_threads}',plan=plan,layout=layout,profile=profile)
              for plan,layout,profile in zip(plans,layouts,profiles)]
    report=plan_loop_workers(loop,strategy,variants,tmp_path/'known')
    assert report['complete_domain'] and report['selected_variant_id']=='cpu-4'
    selected,layout=selected_loop_worker(loop,strategy,report)
    assert selected==four and layout==layouts[1]
    assert report['solver']['global_lower_bound_ticks']==28
    assert not report['target_variants_compiled'] and not report['target_variants_executed']
    assert verify_loop_worker_choice(loop,strategy,report)==report
    tampered=deepcopy(report); tampered['selected_variant_id']='cpu-2'
    with pytest.raises(ValueError,match='complete source replay'): verify_loop_worker_choice(loop,strategy,tampered)
    variants[1]['profile']=None
    incomplete=plan_loop_workers(loop,strategy,variants,tmp_path/'unknown')
    assert not incomplete['complete_domain'] and incomplete['selected_variant_id'] is None
    assert len(incomplete['variants'])==2 and incomplete['solver']['frontier']
    with pytest.raises(ValueError,match='domain unresolved'): selected_loop_worker(loop,strategy,incomplete)

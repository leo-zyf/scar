"""Joint costs preserve scope, raw-clock arithmetic and unvalidated transfer."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scar.compiler.calibrate_native_groups import native_group_statistics, _timer
from scar.ir.execution_recipe import artifact_reference
from scar.ir.implementation_plan import CostEstimate
from scar.ir.record_codec import decode
from scar.optimization.native_group_costs import derive_native_group_costs
from scar.optimization.resource_solver import semantic_sha256


def test_joint_group_mean_and_strict_metadata_binding_without_native(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('metadata adapter must not compile or load native code')
    monkeypatch.setattr('subprocess.Popen', forbidden)
    monkeypatch.setattr('ctypes.CDLL', forbidden)

    def save(name, value):
        path = tmp_path/name
        path.write_text(json.dumps(value))
        return artifact_reference(path, value['schema'])

    plan = dict(schema='scar.stateful-stage-plan', region_sha256='a'*64,
        stages=[dict(id='stage:0', kind='COMPUTE'), dict(id='stage:1', kind='WRITE_SHADOW'),
            dict(id='stage:2', kind='COMMIT')])
    plan_ref = save('plan.json', plan)
    compiler = dict(argv=['inert-metadata-compiler'], identity='not queried', options=[])
    native = dict(schema='scar.native-stateful-artifact', files=dict(plan='plan.json'),
        sha256=dict(plan=plan_ref['sha256']), plan_sha256=semantic_sha256(plan), region_sha256='a'*64,
        build_id='b'*64, compiler=compiler)
    native_ref = save('artifact.json', native)
    identity = dict(build_id=native['build_id'], plan_sha256=native['plan_sha256'], region_sha256='a'*64)
    layout = dict(index=0, stage_indices=[0, 1], begin_cut=0, end_cut=1)
    context = dict(native_identity=identity, source_refs=dict(native_manifest=native_ref, plan=plan_ref),
        stages=[dict(index=i, id=row['id'], kind=row['kind'],
            source_stage_ref=dict(artifact=plan_ref, locator='#/stages/'+str(i))) for i, row in enumerate(plan['stages'])],
        versioned_schedule=dict(tasks=[dict(id=key) for key in
            ('allocate:slot:0', 'stage:0', 'stage:1', 'success:precommit', 'stage:2', 'release:slot:0')]),
        groups=[dict(**layout, original_frame_entry=layout)])
    design = dict(schema='scar.native-stage-group-calibration-design', schema_version=1, timer=_timer(),
        request=dict(native_manifest=native_ref, warmup=2, repetitions=4, array_value=1.25, scalar_value=0.75),
        runtime_context=dict(data_regime='CANONICAL', cache_residency='UNKNOWN'), stage_context=context,
        observer=dict(parent_view=dict(native_manifest=native_ref, plan=plan_ref, identity=identity, compiler=compiler),
            specification=dict(native_manifest=native_ref), groups=[layout], cut_count=2,
            observer_identity='c'*64, retained_residuals=['OBSERVER_CLOCKS_AND_CONSTRAINTS']))
    reference = save('design.json', design)
    observed = dict(schema='scar.native-stage-group-calibration', schema_version=1, design=reference,
        frozen_design_sha256=reference['sha256'], clock_overhead_subtracted=False,
        full_native_prefix_suffix_and_commits_executed=True, warmup=2, repetitions=4,
        group_layout=[layout], residuals=['CANONICAL_DATA_TRANSFER'], samples=[
            dict(ordinal=i, native_status=0, ffi_start_ns=100+100*i, ffi_end_ns=190+100*i,
                cuts_ns=[120+100*i, 120+100*i+duration]) for i, duration in enumerate((2, 3, 3, 2))])
    observed['statistics'] = native_group_statistics(observed)
    result = derive_native_group_costs(observed, design)
    group, = result['cost_groups']; cost = decode(CostEstimate, group['cost'])
    assert group['task_ids'] == ['stage:0', 'stage:1'] and group['jointly_charged_once']
    assert cost.estimate_ns == 2 and cost.basis == 'DECLARED_MODEL'
    assert cost.lower_ns is cost.upper_ns is None and cost.missing_terms
    assert group['statistics']['exact_mean_ns'] == dict(numerator=10, denominator=4)
    assert group['statistics']['integer_rounding'] == 'NEAREST_INTEGER_NANOSECOND_TIES_TO_EVEN'
    assert group['statistics']['warmup_calls_excluded'] == 2 and group['statistics']['measurement_samples'] == 4
    assert result['whole_method_estimate_ns'] is None and result['entry_path_protocol']['status'] == 'UNKNOWN'
    assert not result['ffi_window_imported_as_cost'] and not result['outside_group_complement_imported_as_cost']
    longer = deepcopy(observed)
    for sample in longer['samples']: sample['ffi_end_ns'] += 10000
    longer['statistics'] = native_group_statistics(longer)
    assert derive_native_group_costs(longer, design)['cost_groups'][0]['cost']['estimate_ns'] == 2
    forged = deepcopy(observed); forged['statistics']['groups'][0]['exact_mean_ns']['numerator'] += 1
    with pytest.raises(ValueError, match='statistics'):
        derive_native_group_costs(save('rehashed-observations.json', forged), design)
    bad_clock = deepcopy(observed); bad_clock['samples'][0]['cuts_ns'] = [150, 149]
    with pytest.raises(ValueError, match='clock'):
        derive_native_group_costs(bad_clock, design)

    def rebind(changed):
        ref = save('changed-design.json', changed)
        document = deepcopy(observed)
        document.update(design=ref, frozen_design_sha256=ref['sha256'])
        return document

    wrong = deepcopy(design); wrong['timer']['inner'] = 'WHOLE_METHOD'
    with pytest.raises(ValueError, match='timer'):
        derive_native_group_costs(rebind(wrong), wrong)
    wrong = deepcopy(design); wrong['observer']['parent_view']['identity']['plan_sha256'] = '0'*64
    with pytest.raises(ValueError, match='plan'):
        derive_native_group_costs(rebind(wrong), wrong)
    wrong = deepcopy(design); wrong['observer']['groups'][0]['stage_indices'] = [0, 2]
    document = rebind(wrong); document['group_layout'] = wrong['observer']['groups']
    document['statistics'] = native_group_statistics(document)
    with pytest.raises(ValueError, match='contiguous'):
        derive_native_group_costs(document, wrong)

    # Exercise the retained real records when available; no cold source replay,
    # target imports, compiler identity query or ELF loading occurs here.
    real = Path(__file__).resolve().parents[2]/'artifacts/current/experiments/native-frame-groups-v1'
    if (real/'observations.json').is_file():
        actual = derive_native_group_costs(real/'observations.json', real/'design.json')
        joint, = actual['cost_groups']
        assert joint['task_ids'] == ['stage:'+str(i) for i in range(9)]
        assert joint['statistics']['exact_mean_ns'] == dict(numerator=3781846, denominator=7)
        assert decode(CostEstimate, joint['cost']).estimate_ns == 540264

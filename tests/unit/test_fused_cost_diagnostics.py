"""Residual replay cannot promote inspected/rejected models into costs."""
from copy import deepcopy

import pytest

from scar.codegen.cpu_fused_cost import fused_features
from scar.optimization.fused_cost_diagnostics import analyze_fused_profile


def test_residual_replay_groups_failures_preserves_coefficients_and_admission():
    from scar.codegen.cpu_topology_ablation import _cases, _layout
    cases = _cases()
    assert [case['dependent_operation_depth'] for case in cases] == [8, 5, 6]
    for case in cases:
        assert case['arithmetic'] == dict(add=4, sub=2, mul=2, div=0, neg=0)
        graph = {node['id']: node for node in case['graph']}; reached = set()
        def visit(key):
            if key in reached: return
            reached.add(key)
            if key in graph:
                for operand in graph[key]['inputs']: visit(operand)
        visit('v7')
        assert set(graph) <= reached and {'x0', 'x1', 'x2', 'x3'} <= reached
    for groups in (1, 2):
        even = _layout(384, groups, 'ALIGNED_EVEN_STRIDE')
        shifted = _layout(384, groups, 'SHIFTED_ODD_STRIDE')
        assert len({row['storage_group'] for row in even}) == groups
        assert all(row['stride'] % 2 == row['offset'] % 2 == 0 for row in even)
        assert all(row['stride'] % 2 == 1 for row in shifted)
    p = dict(rows=4, cols=4, workers=2, arithmetic=dict(add=1, sub=0, mul=0, div=0, neg=0),
        array_loads=1, output_stores=1, dependent_operation_depth=1,
        operand_layouts=[dict(stride=4, offset=0, storage_group=0)], backing_storage_groups=1)
    features, geometry = fused_features(p, 4096)
    coefficients = {key: 100. if key == 'stage' else 0. for key in features}
    observations = [dict(primitive='fixture_' + str(i), geometry='INDEPENDENT', family='DEPENDENT_CHAIN',
        parameters=deepcopy(p), features=deepcopy(features), geometry_properties=deepcopy(geometry),
        holdout=True, median_ns=median, samples_ns=[median-1, median, median+1])
        for i, median in enumerate((50., 75.))]
    held = [dict(primitive=row['primitive'], geometry=row['geometry'], rows=4, cols=4,
        observed_ns=row['median_ns'], predicted_ns=100.,
        relative_error=abs(100-row['median_ns'])/row['median_ns']) for row in observations]
    model = dict(schema='scar.cpu-fused-cost-model', schema_version=2, features=list(features),
        observations=observations, cache_geometry=dict(reported_capacity_bytes=4096),
        local_models={'fixture-regime': dict(coefficients=coefficients, holdout=held,
            precision_policy=dict(maximum_holdout_relative_error=.2), admitted=False,
            maximum_holdout_relative_error=1., rank=1, active_parameters=1,
            residual=dict(rmse_ns=1., max_relative_error=.1))})
    original = deepcopy(model)
    receipt = dict(uri='/original/profile.json', sha256='0'*64, size_bytes=1,
        schema='scar.cpu-fused-cost-model')
    report = analyze_fused_profile(model, receipt)
    local = report['local_models'][0]
    assert model == original
    assert local['summary']['overpredictions'] == 2
    assert local['summary']['maximum_relative_error'] == 1.
    assert local['summary']['predictions_outside_observed_repeat_range'] == 2
    assert not local['original_admitted'] and not report['coefficients_refitted']
    assert not report['selectable'] and not report['exportable_as_complete_cost']
    assert not report['empirical_error_is_theoretical_bound']
    assert report['feature_collisions']['bucket_count'] == 1
    assert report['feature_collisions']['largest'][0]['repeat_ranges_disjoint']
    assert {row['dimension'] for row in local['grouped_residuals']} >= {
        'geometry', 'arithmetic', 'dependency_depth', 'cache_extent', 'backing_storage_groups'}
    altered = deepcopy(model); altered['local_models']['fixture-regime']['holdout'][0]['predicted_ns'] = 99.
    with pytest.raises(ValueError, match='coefficient replay'):
        analyze_fused_profile(altered, receipt)
    altered = deepcopy(model); altered['local_models']['fixture-regime']['admitted'] = True
    with pytest.raises(ValueError, match='precision admission'):
        analyze_fused_profile(altered, receipt)

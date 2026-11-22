"""Cost gates keep deployment scopes and direct/component evidence separate."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import encode
from scar.optimization.deployment_cost import plan_deployment, verify_deployment_plan


def profile(tmp_path, name, boundary, mode, values, *, basis='OBSERVED', count=1):
    context = dict(boundary=boundary, workload_id='original-loop-sha', artifact_id='native-build-sha',
        cpu_ids=[0, 1], required_memory_bytes=64, memory_scope='INCREMENTAL_ARRAY_PAYLOAD',
        external_assumptions=['FP_STATUS_FLAGS_NOT_USED_AS_PROGRAM_OUTPUT'],
        runtime_guards=['EXACT_INPUT_SHAPE', 'CPU_ALLOCATION_AVAILABLE'])
    scope = 'DIRECT_COMPONENT_TIMING' if mode == 'COMPONENTS' else 'DIRECT_COMPLETE_SCOPE_TIMING'
    records = {key: dict(value=value, unit='seconds', basis=basis, measurement_scope=scope,
                        **({'reuse_count': count} if mode == 'COMPLETE_SCOPE' else {}))
               for key, value in values.items()}
    original = dict(schema='scar.test-deployment-observations', context=context, costs=records)
    path = tmp_path / (name + '-evidence.json'); path.write_text(json.dumps(original))
    raw = path.read_bytes()
    artifact = ProviderArtifact(str(path), sha256(raw).hexdigest(), len(raw), original['schema'])
    def reference(pointer): return encode(ProviderFactReference(artifact, pointer))
    document = dict(schema='scar.deployment-cost-profile', schema_version=1, context=context,
        mode=mode, context_evidence=[reference('#/context')], costs={key: dict(value=value,
        unit='seconds', basis=basis, evidence=[reference('#/costs/' + key)]) for key, value in values.items()})
    if mode == 'COMPLETE_SCOPE': document['observed_reuse_count'] = count
    return document, context


def test_component_break_even_is_exact_sourced_and_can_gate_before_loading(tmp_path):
    document, context = profile(tmp_path, 'components', 'FRESH_PROCESS', 'COMPONENTS',
        dict(original_one_time=0, native_load=0.9, native_setup=0.1,
             original_per_call=0.1, native_per_call=0.05), basis='DECLARED_MODEL')
    request = dict(context=context, available_cpu_ids=[0, 1, 2], memory_budget_bytes=64)
    cold = plan_deployment(document, reuse_count=1, **request)
    assert cold['decision'] == 'NO_OP' and cold['break_even_reuse_count'] == 21
    assert cold['qualifying_reuse_range'] == dict(minimum=21, maximum=None)
    repeated = plan_deployment(document, reuse_count=23, **request)
    assert repeated['decision'] == 'LOAD_NATIVE_CANDIDATE'
    assert repeated['total_costs']['original']['seconds'] == 2.3
    assert repeated['total_costs']['native']['seconds'] == 2.15
    assert repeated['status'] == 'COST_MODEL_EVALUATED' and not repeated['whole_scope_cost_observed']
    assert repeated['selection_before_heavy_load']['requires_heavy_load'] is False
    assert not repeated['deployment_authorized'] and not repeated['target_variants_executed']
    assert repeated['external_assumptions'] == context['external_assumptions']
    assert verify_deployment_plan(repeated) == repeated
    missing = deepcopy(document); del missing['costs']['native_load']
    unresolved = plan_deployment(missing, reuse_count=23, **request)
    assert unresolved['decision'] == 'NO_OP' and unresolved['total_costs'] is None
    assert any(row['field'] == 'costs.native_load' for row in unresolved['frontier'])
    low = plan_deployment(document, reuse_count=23, **{**request, 'memory_budget_bytes': 63})
    assert low['decision'] == 'NO_OP' and 'declared_memory_budget_exceeded' in low['reasons']
    changed = deepcopy(repeated); changed['break_even_reuse_count'] = 22
    with pytest.raises(ValueError, match='scope/evidence replay'):
        verify_deployment_plan(changed)


def test_complete_boundary_is_replayed_at_its_original_count_without_setup_subtraction(tmp_path):
    warm, context = profile(tmp_path, 'warm', 'WARM_SESSION', 'COMPLETE_SCOPE',
                            dict(original_total=25, native_total=1))
    request = dict(context=context, reuse_count=1, available_cpu_ids=[0, 1], memory_budget_bytes=64)
    path = tmp_path / 'profile.json'; path.write_text(json.dumps(warm))
    raw = path.read_bytes()
    reference = ProviderFactReference(ProviderArtifact(str(path), sha256(raw).hexdigest(),
        len(raw), warm['schema']), '#')
    report = plan_deployment(reference, **request)
    assert report['decision'] == 'LOAD_NATIVE_CANDIDATE' and report['whole_scope_cost_observed']
    assert report['total_costs']['original']['seconds'] == 25
    assert report['break_even_reuse_count'] is None and report['qualifying_reuse_range'] is None
    assert report['setup_inferred_from_scope_difference'] is False
    assert verify_deployment_plan(report) == report
    mismatch = plan_deployment(warm, **{**request, 'context': {**context, 'boundary': 'FRESH_PROCESS'}})
    assert mismatch['decision'] == 'NO_OP' and any(row['field'] == 'context' for row in mismatch['frontier'])
    extrapolated = plan_deployment(warm, **{**request, 'reuse_count': 2})
    assert extrapolated['decision'] == 'NO_OP' and extrapolated['total_costs'] is None
    forged = deepcopy(warm); forged['observed_reuse_count'] = 2
    invalid = plan_deployment(forged, **{**request, 'reuse_count': 2})
    assert invalid['decision'] == 'NO_OP' and any('source reuse count differs' in row['reason'] for row in invalid['frontier'])
    bad = deepcopy(warm); bad['costs']['native_total']['evidence'] = warm['costs']['original_total']['evidence']
    invalid = plan_deployment(bad, **request)
    assert invalid['decision'] == 'NO_OP' and any('source value/unit/basis differs' in row['reason'] for row in invalid['frontier'])


def test_default_gain_is_strict_and_legacy_deployment_receipt_replays(tmp_path):
    document, context = profile(tmp_path, 'boundary', 'FRESH_PROCESS', 'COMPONENTS',
        dict(original_one_time=0, native_load=9, native_setup=0, original_per_call=100, native_per_call=90),
        basis='DECLARED_MODEL')
    request = dict(context=context, available_cpu_ids=[0, 1], memory_budget_bytes=64)
    boundary = plan_deployment(document, reuse_count=1, **request)
    assert boundary['request']['minimum_gain'] == 0.01
    assert boundary['request']['strict_minimum_gain'] is True
    assert boundary['decision'] == 'NO_OP' and boundary['break_even_reuse_count'] == 2
    assert verify_deployment_plan(boundary) == boundary
    assert plan_deployment(document, reuse_count=2, **request)['decision'] == 'LOAD_NATIVE_CANDIDATE'
    legacy = plan_deployment(document, reuse_count=1, minimum_gain=0.01, strict_minimum_gain=False, **request)
    assert 'strict_minimum_gain' not in legacy['request']
    assert legacy['decision'] == 'LOAD_NATIVE_CANDIDATE' and legacy['break_even_reuse_count'] == 1
    assert verify_deployment_plan(legacy) == legacy

"""Automatic entry costs cannot inherit a manual installer observation scope."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

from scar.optimization.deployment_cost import plan_deployment
from scar.optimization.deployment_gate import deployment_gate, verify_deployment_gate


def setup_gate(tmp_path, *, legacy=False, components=False, launch_policy='COST_GATED_DEPLOYMENT', argv=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    entry = tmp_path / 'entry.py'; entry.write_text("raise RuntimeError('must not execute')\n")
    ids = dict(zip(('build_id', 'loop_sha256', 'strategy_sha256', 'execution_plan_sha256', 'contract_sha256'),
                   (letter * 64 for letter in 'abcde')))
    scope = dict(kind='AUTOMATIC_RUNNER', entry_path=str(entry), entry_sha256=sha256(entry.read_bytes()).hexdigest(),
        argv=[str(entry), '--out', str(tmp_path / 'output.npy')] if argv is None else argv,
        runner_implementation_sha256='f' * 64)
    actual = dict(bundle_ids=ids, admission_sha256='1' * 64, entry_scope=scope,
                  reuse_count=1, available_cpu_ids=[0, 1], memory_budget_bytes=64)
    context = dict(boundary='FRESH_PROCESS', workload_id='loop', artifact_id=ids['build_id'],
        cpu_ids=[0, 1], required_memory_bytes=64, memory_scope='INCREMENTAL_ARRAY_PAYLOAD',
        external_assumptions=['FP_STATUS_FLAGS_NOT_USED_AS_PROGRAM_OUTPUT'], runtime_guards=['SOURCE_AND_SHAPE'])
    if not legacy:
        context.update(runtime_bundle=dict(ids=ids, admission_sha256=actual['admission_sha256']), entry_scope=scope,
                       launch_policy=launch_policy)
    basis = 'DECLARED_MODEL' if components else 'OBSERVED'
    mode = 'COMPONENTS' if components else 'COMPLETE_SCOPE'
    values = (dict(original_one_time=0, native_load=0.1, native_setup=0.1, original_per_call=2, native_per_call=0.5)
              if components else dict(original_total=2, native_total=0.5))
    source = dict(schema='scar.test-deployment-gate-facts', context=context, costs={key:dict(value=value,
        unit='seconds', basis=basis, measurement_scope='DIRECT_COMPLETE_SCOPE_TIMING', reuse_count=1)
        for key, value in values.items()})
    evidence = tmp_path / 'evidence.json'; evidence.write_text(json.dumps(source)); raw = evidence.read_bytes()
    artifact = dict(uri=str(evidence), sha256=sha256(raw).hexdigest(), size_bytes=len(raw), schema=source['schema'])
    def reference(pointer): return dict(artifact=artifact, locator=pointer)
    profile = dict(schema='scar.deployment-cost-profile', schema_version=1, mode=mode, context=context,
        context_evidence=[reference('#/context')], costs={key:dict(value=value, unit='seconds', basis=basis,
            evidence=[reference('#/costs/' + key)]) for key, value in values.items()})
    if not components: profile['observed_reuse_count'] = 1
    report = plan_deployment(profile, context=context, reuse_count=1, available_cpu_ids=[0, 1], memory_budget_bytes=64)
    path = tmp_path / 'plan.json'; path.write_text(json.dumps(report))
    return path, actual, profile


def test_exact_automatic_scope_replays_before_native_load_without_heavy_imports(tmp_path):
    path, actual, profile = setup_gate(tmp_path)
    result = deployment_gate(path, **actual, profile=profile)
    assert result['decision'] == 'LOAD_NATIVE_CANDIDATE' and result['selected']
    assert not result['native_library_loaded'] and not result['target_executed'] and not result['deployment_authorized']
    assert result['whole_scope_cost_observed'] and not result['frontier']
    assert verify_deployment_gate(path, result, **actual, profile=profile) == result
    changed = deepcopy(actual); changed['entry_scope']['argv'][-1] += '.different'
    declined = deployment_gate(path, **changed)
    assert declined['decision'] == 'NO_OP' and any(row['field'] == 'context.entry_scope' for row in declined['frontier'])
    changed = deepcopy(actual); changed['bundle_ids']['build_id'] = '2' * 64
    assert deployment_gate(path, **changed)['decision'] == 'NO_OP'
    # This subprocess only imports metadata arithmetic. The fixture entry raises
    # if executed, and no target or native library is run in this test.
    code = (f"import sys; sys.path.insert(0,{str(Path(__file__).resolve().parents[2])!r}); "
            "import scar.optimization.deployment_gate; "
            "assert not any(name.startswith(('scar.compiler','scar.codegen','scar.ir','numpy','ctypes')) "
            "for name in sys.modules)")
    subprocess.run([sys.executable, '-I', '-S', '-c', code], check=True, capture_output=True, text=True, timeout=10)
    Path(actual['entry_scope']['entry_path']).write_text('changed entry\n')
    assert any(row['field'] == 'entry_scope.entry_path' for row in deployment_gate(path, **actual)['frontier'])


def test_legacy_scope_reuse_resources_and_models_keep_fail_closed_contracts(tmp_path):
    path, actual, _ = setup_gate(tmp_path / 'legacy', legacy=True)
    legacy = deployment_gate(path, **actual)
    assert legacy['decision'] == 'NO_OP'
    assert {row['field'] for row in legacy['frontier']} >= {'context.runtime_bundle', 'context.entry_scope'}
    path, actual, profile = setup_gate(tmp_path / 'model', components=True)
    model = deployment_gate(path, **actual)
    assert model['decision'] == 'LOAD_NATIVE_CANDIDATE' and not model['whole_scope_cost_observed']
    assert set(model['cost_bases'].values()) == {'DECLARED_MODEL'}
    assert model['model_assumptions'] and model['external_assumptions'] == profile['context']['external_assumptions']
    bypass_path, bypass_actual, _ = setup_gate(tmp_path / 'measure', launch_policy='MEASURE_CANDIDATE')
    bypass = deployment_gate(bypass_path, **bypass_actual)
    assert bypass['decision'] == 'NO_OP' and any(row['field'] == 'context.launch_policy' for row in bypass['frontier'])
    empty_path, empty_actual, _ = setup_gate(tmp_path / 'empty', argv=[])
    assert deployment_gate(empty_path, **empty_actual)['decision'] == 'LOAD_NATIVE_CANDIDATE'
    count = deployment_gate(path, **{**actual, 'reuse_count': 2})
    assert count['decision'] == 'NO_OP' and any(row['field'] == 'reuse_count' for row in count['frontier'])
    cpu = deployment_gate(path, **{**actual, 'available_cpu_ids': [0]})
    assert cpu['decision'] == 'NO_OP' and 'required_cpu_allocation_unavailable' in cpu['reasons']
    memory = deployment_gate(path, **{**actual, 'memory_budget_bytes': 63})
    assert memory['decision'] == 'NO_OP' and 'declared_memory_budget_exceeded' in memory['reasons']
    forged = json.loads(path.read_bytes()); forged['total_costs']['native']['seconds'] = 0
    invalid = deployment_gate(forged, **actual)
    assert invalid['decision'] == 'NO_OP' and invalid['frontier'][0]['field'] == 'deployment_plan'

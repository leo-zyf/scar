"""CPU choices use retained calibrated models and preserve the original call."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json

import pytest

from test_cpu_hardware_placement import hardware, plan
from test_region_cost_model import setup_model
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import encode
from scar.optimization import optimize_region, bind_cpu_placement, propose_cpu_placements
from scar.optimization.cpu_choices import plan_cpu_choices, read_cpu_choice_prediction, verify_cpu_choice_report
from scar.optimization.schedule import build_region_schedule
from scar.optimization.resource_solver import semantic_sha256


def setup_choices(hardware, tmp_path):
    """Two complete retained primitive-profile contexts for independent tests."""
    seed = tmp_path / 'seed'; seed.mkdir()
    _, _, seed_profile, seed_source = setup_model(hardware, seed)
    base = optimize_region(plan().region, memory_budget_bytes=4096)
    variants = []
    for placement in propose_cpu_placements(base, hardware, max_threads=2)[:2]:
        bound = bind_cpu_placement(base, placement)
        profile = deepcopy(seed_profile); raw = json.loads(seed_source.read_bytes())
        target = dict(target_sha256=placement.target_sha256, executor_id=placement.executor_id,
                      cpu_ids=list(placement.cpu_ids))
        options = ['-fixture'] + (['-fopenmp'] if bound.cpu_threads > 1 else [])
        intercept = 3 if bound.cpu_threads > 1 else 20
        profile['target'] = raw['target'] = target
        profile['backend']['compiler']['options'] = raw['compiler']['options'] = options
        profile['model']['task_models']['COMPUTE']['intercept']['value'] = intercept
        raw['fits']['COMPUTE']['intercept']['value'] = intercept
        source = tmp_path / f'cpu-{bound.cpu_threads}-calibration.json'
        source.write_text(json.dumps(raw)); data = source.read_bytes()
        artifact = encode(ProviderArtifact(str(source), sha256(data).hexdigest(), len(data), raw['schema']))

        def rewrite(value):
            if isinstance(value, dict):
                if set(value) == {'artifact', 'locator'}:
                    value['artifact'] = deepcopy(artifact)
                else:
                    for child in value.values(): rewrite(child)
            elif isinstance(value, list):
                for child in value: rewrite(child)

        rewrite(profile)
        variants.append(dict(id=f'cpu-{bound.cpu_threads}', plan=bound,
                             schedule=build_region_schedule(bound, hardware), profile=profile))
    return base, variants


def test_cpu_family_selection_and_pure_replay_preserve_all_receipts(hardware, tmp_path):
    base, variants = setup_choices(hardware, tmp_path)
    out = tmp_path / 'choice'
    result = plan_cpu_choices(base, variants, out, input_shapes=(4, 5))
    assert result['complete_domain'] and result['selected_variant'] == {'id': 'cpu-2', 'index': 1}
    assert result['selected_bound_solver_result']['solution']['makespan_ticks'] == 48
    assert 'base_family' not in result['semantic_contract']
    assert len(result['semantic_contract']['base_family_sha256']) == 64
    assert result['variants'][1]['prediction']['locator'] == '#/prediction'
    assert read_cpu_choice_prediction(result['variants'][1])['prediction_available']
    wrong_pointer = deepcopy(result['variants'][1]); wrong_pointer['prediction']['locator'] = '#/legality'
    with pytest.raises(ValueError, match='original variant evidence'):
        read_cpu_choice_prediction(wrong_pointer)
    assert result['original_python_no_op'] == {'available': True, 'cost_modeled': False, 'outperformed': False}
    assert not result['target_variants_compiled'] and not result['target_variants_executed']
    assert verify_cpu_choice_report(out / 'choice.json')['choice_id'] == result['choice_id']
    altered = deepcopy(result); altered['selected_variant'] = {'id': 'cpu-1', 'index': 0}
    with pytest.raises(ValueError, match='sourced replay'):
        verify_cpu_choice_report(altered)
    domain_path = out / 'domain.json'; original_domain = domain_path.read_bytes()
    tampered_domain = json.loads(original_domain)
    tampered_domain['semantic_contract']['output_bytes'] += 8
    changed_bytes = (json.dumps(tampered_domain, indent=2) + '\n').encode(); domain_path.write_bytes(changed_bytes)
    resigned = deepcopy(result)
    resigned['domain'].update(sha256=sha256(changed_bytes).hexdigest(), bytes=len(changed_bytes),
                              semantic_sha256=semantic_sha256(tampered_domain))
    with pytest.raises(ValueError, match='family contract differs'):
        verify_cpu_choice_report(resigned)
    domain_path.write_bytes(original_domain)
    source = result['variants'][0]['plan']['path']
    with open(source, 'a') as stream: stream.write(' ')
    with pytest.raises(ValueError, match='source receipt'):
        verify_cpu_choice_report(out / 'choice.json')


def test_missing_profile_and_non_cpu_contract_changes_remain_domain_frontiers(hardware, tmp_path):
    base, variants = setup_choices(hardware, tmp_path)
    missing = [variants[0], {**variants[1], 'profile': None, 'frontier_reason': 'missing CPU-context profile'}]
    result = plan_cpu_choices(base, missing, tmp_path / 'missing', input_shapes=(4, 5))
    assert not result['complete_domain'] and result['solver']['frontier']
    assert result['solver']['status'] == 'UNRESOLVED_OPTION_DOMAIN'
    verify_cpu_choice_report(tmp_path / 'missing/choice.json')
    changed = dict(variants[1]); original = changed['plan']
    changed['plan'] = replace(original, buffer_policy='input_alias')
    rejected = plan_cpu_choices(base, [variants[0], changed], tmp_path / 'changed', input_shapes=(4, 5))
    assert not rejected['complete_domain'] and rejected['variants'][1]['family_issues']
    assert rejected['selected_variant']['id'] == 'cpu-1'
    with pytest.raises(ValueError, match='unsupported CPU variant constraints'):
        plan_cpu_choices(base, [{**variants[0], 'requires_async_copy': True}],
                         tmp_path / 'unsupported', input_shapes=(4, 5))

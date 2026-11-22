"""Whole-process delivery authority is scoped to the selected native bytes."""

import json

from scar.compiler.entry_delivery import BOUNDARY, STATEFUL, replay_entry_delivery
from scar.ir.execution_recipe import artifact_reference
from scar.optimization.paired_entry_selection import compare_entry_pairs, replay_entry_pairs


def test_original_and_unmatched_winner_do_not_load_candidate():
    candidate = {'id': 'not-the-winner', 'route': 'STATEFUL_CAPSULE', 'artifact': None}
    original = replay_entry_delivery({}, selected_id='original', candidate=candidate)
    unmatched = replay_entry_delivery({}, selected_id='other-winner', candidate=candidate)
    assert original['final_action'] == unmatched['final_action'] == 'KEEP_ORIGINAL'
    assert original['qualification'] is unmatched['qualification'] is None
    assert not original['deployment_authorized'] and not unmatched['deployment_authorized']


def test_unmatched_source_and_untrusted_receipt_keep_original(tmp_path):
    entry = tmp_path / 'entry.py'
    entry.write_text('print(1)\n')
    source = artifact_reference(entry, 'scar.python-source')
    scope = dict(boundary=BOUNDARY, entry=str(entry), argv=['1'], entry_sha256=source['sha256'],
                 target_sha256='0' * 64, memory_budget_bytes=100)
    capsule = tmp_path / 'payload.json'
    payload = dict(schema=STATEFUL, schema_version=1, deployment_authorized=True,
                   entry_context=dict(entry_path=str(entry), entry_sha256=source['sha256'], argv=['2']))
    capsule.write_text(json.dumps(payload))
    candidate = dict(id='native', route='STATEFUL_CAPSULE', artifact=artifact_reference(capsule, STATEFUL))
    rejected = replay_entry_delivery(scope, selected_id='native', candidate=candidate)
    assert rejected['final_action'] == 'KEEP_ORIGINAL'
    assert rejected['qualification']['status'] == 'UNRESOLVED'
    assert 'entry source or arguments differ' in rejected['reason']
    assert rejected['validation'] is None and rejected['launch_authority'] is None

    capsule.write_text(json.dumps(payload, indent=2))
    rejected = replay_entry_delivery(scope, selected_id='native', candidate=candidate)
    assert 'byte identity differs' in rejected['reason']
    assert not rejected['deployment_authorized']


def test_strict_paired_gain_boundary_keeps_legacy_replay():
    def runs(candidate):
        return [dict(pair=pair, mode=mode, wall_ns=wall, returncode=0,
                     stdout_sha256='0' * 64, stderr_sha256='1' * 64)
                for pair in range(3) for mode, wall in (('original', 100), ('native', candidate))]

    scope = dict(incumbent='original', challenger='native', warmup_pairs=0, measurement_pairs=3)
    exact = compare_entry_pairs(runs(99), **scope)
    assert exact['policy']['strict_minimum_gain'] is True
    assert exact['verdict'] == 'KEEP_INCUMBENT'
    assert replay_entry_pairs(runs(99), exact, **scope) == exact
    assert compare_entry_pairs(runs(98), **scope)['verdict'] == 'ADVANCE_TO_RELEASE_REVIEW'
    legacy = compare_entry_pairs(runs(95), minimum_gain=0.05, strict_minimum_gain=False, **scope)
    assert 'strict_minimum_gain' not in legacy['policy']
    assert legacy['verdict'] == 'ADVANCE_TO_RELEASE_REVIEW'
    assert replay_entry_pairs(runs(95), legacy, **scope) == legacy

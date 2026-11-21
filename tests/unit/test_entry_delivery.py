"""Whole-process delivery authority is scoped to the selected native bytes."""

import json

from scar.compiler.entry_delivery import BOUNDARY, STATEFUL, replay_entry_delivery
from scar.ir.execution_recipe import artifact_reference


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

"""Whole-process delivery authority is scoped to the selected native bytes."""

import json
import os
from pathlib import Path
from stat import S_IMODE

import pytest

from scar.codegen import cpu_native_route_package as route_package
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


@pytest.mark.parametrize('fixed_grid', [False, True], ids=['array', 'fixed-grid'])
def test_scoped_native_route_package_is_published_and_replayed_by_reference(tmp_path, monkeypatch, fixed_grid):
    schema = route_package.FIXED_GRID_SCHEMA if fixed_grid else route_package.ARRAY_SCHEMA
    project = tmp_path / 'project'
    project.mkdir()
    plan = tmp_path / 'plan.json'
    plan.write_text('{"schema":"scar.test-route-plan"}\n')
    plan_ref = artifact_reference(plan, 'scar.test-route-plan')
    output = tmp_path / 'output'
    scope = dict(argv=['--output', str(output)], cpu_ids=[min(os.sched_getaffinity(0))],
                 hardware={'target': 'host'}, source_environment={'PATH': '/usr/bin:/bin'})
    report = dict(schema=schema, schema_version=1, plan=plan_ref, status='ADMITTED',
                  routes={'selected': {'project_root': str(project)}}, **scope)
    if fixed_grid:
        report.update(selected_runtime='PROVIDER_MAP', publication={'path': str(output)})
        derive = '_derive_fixed_grid'
        export = route_package.export_cpu_fixed_grid_native_route_package
        verify = route_package.verify_cpu_fixed_grid_native_route_package
        prepare = route_package.prepare_cpu_fixed_grid_native_route_launch
        launch_options = {}
        publication_refusal = 'FULL_INPUT_LIFETIME_STABILITY_REQUIRED'
    else:
        derive = 'derive_cpu_array_native_route_package'
        export = route_package.export_cpu_array_native_route_package
        verify = route_package.verify_cpu_array_native_route_package
        prepare = route_package.prepare_cpu_array_native_route_launch
        launch_options = {'fresh_root': project / 'output'}
        publication_refusal = 'FRESH_OUTPUT_DIRECTORY_REQUIRED'

    def derive_package(reference):
        assert reference == plan_ref
        return report

    monkeypatch.setattr(route_package, derive, derive_package)
    destination = tmp_path / 'delivery'
    ref = export(plan_ref, destination)
    member = Path(ref['uri'])
    assert tuple(destination.iterdir()) == (member,)
    assert S_IMODE(member.stat().st_mode) == 0o400
    assert member.read_text() == json.dumps(report, sort_keys=True, indent=2) + '\n'
    assert verify(ref) == report
    with pytest.raises(ValueError, match='fresh'):
        export(plan_ref, project / 'package')
    with pytest.raises(ValueError, match='fresh'):
        export(plan_ref, destination)

    decision = prepare(ref, argv=scope['argv'], cpu_ids=scope['cpu_ids'], target=scope['hardware'],
                       environment={}, **launch_options)
    assert decision['reasons'] == ['ENVIRONMENT_DOES_NOT_MATCH', publication_refusal]
    assert decision['execution'] is None and decision['target_executed'] is False

    report['status'] = 'CHANGED'
    with pytest.raises(ValueError, match='differs .*replay'):
        verify(ref)
    member.chmod(0o600)
    member.write_text(member.read_text() + '\n')
    with pytest.raises(ValueError, match='pinned file changed'):
        verify(ref)

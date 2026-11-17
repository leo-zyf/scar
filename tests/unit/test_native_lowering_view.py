"""Actual-renderer source maps need no native compiler, loader or target."""
from dataclasses import replace
from hashlib import sha256
import ctypes
import json
import subprocess

import pytest

from test_cpu_hardware_placement import hardware
from test_stateful_stage_dag import _inputs, _region, _render
from scar.codegen import stateful
from scar.codegen.stateful_stage_capture import capture_stateful_stage_sources
from scar.compiler.model import Expr
from scar.compiler.native_lowering_view import derive_stateful_lowering_view
from scar.compiler.stateful_model import StatefulAction, document, stateful_digest
from scar.ir.execution_recipe import artifact_reference


def test_actual_source_map_keeps_ambiguous_bodies_and_rejects_changed_members(hardware, tmp_path, monkeypatch):
    region = _region((*_inputs(), Expr('factor', 'neg', ('s',))), ('factor',))
    region = replace(region, actions=region.actions+tuple(
        StatefulAction('write:'+str(i), 'write', 'factor', 'left', None, i, i+1,
            'assign', {'port':'world'+str(i)}, {'port':'world'+str(i+1)}) for i in range(2)))
    plan, _, binding, _ = _render(region, hardware)

    def forbidden(*args, **kwargs):
        raise AssertionError('source-view test must not compile, load native, or execute a target')

    for owner, name in ((subprocess, 'Popen'), (ctypes, 'CDLL'), (stateful, 'load'),
            (stateful, 'build'), (stateful, '_providers'), (stateful, '_admit')):
        monkeypatch.setattr(owner, name, forbidden)
    core = dict(region_sha256=stateful_digest(region), plan_sha256=stateful._digest(plan),
        provider_runtime={'fixture':'METADATA_ONLY_NOT_PROVIDER_ADMISSION'})
    files = dict(region='region.json', plan='plan.json', source='stateful.cpp', library='unloaded.so',
        **{'stateful_abi_v1.h':'stateful_abi_v1.h', 'scar_abi_v1.h':'scar_abi_v1.h'})
    data = dict(region=json.dumps(document(region)).encode(), plan=json.dumps(plan).encode(),
        source=stateful._source(region, plan, binding, core).encode(),
        library=b'NOT_AN_EXECUTABLE_METADATA_FIXTURE')
    for name in ('stateful_abi_v1.h', 'scar_abi_v1.h'):
        data[name] = (stateful._HERE/name).read_bytes()
    for role, name in files.items():
        (tmp_path/name).write_bytes(data[role])
    hashes = {role:sha256(raw).hexdigest() for role, raw in data.items()}
    compiler = dict(argv=['unexecuted-fixture-g++'], identity='NOT_QUERIED', options=['-std=c++20', '-O3'])
    producer = '0'*64  # Historical producer stamp is retained, never replaced by today's stamp.
    build_id = stateful._digest(dict(source=hashes['source'],
        headers={name:hashes[name] for name in ('stateful_abi_v1.h', 'scar_abi_v1.h')},
        compiler=compiler['argv'], identity=compiler['identity'], options=compiler['options'],
        implementation_sha256=producer, contract=core))
    manifest = dict(schema=stateful.SCHEMA, schema_version=1, stateful_abi_version=1,
        files=files, sha256=hashes, **core, contract_sha256=stateful._digest(core),
        build_id=build_id, implementation_sha256=producer, compiler=compiler,
        cpu_ids=list(binding.cpu_ids), external_assumptions=plan['external_assumptions'],
        **{key:plan[key] for key in ('cpu_threads', 'external_array_bytes', 'workspace_payload_bytes',
            'total_region_array_payload_bytes', 'memory_budget_bytes')})
    path = tmp_path/'artifact.json'; path.write_text(json.dumps(manifest))
    capture = capture_stateful_stage_sources(region, plan, binding, core)
    capture_path = tmp_path/'capture.json'; capture_path.write_text(json.dumps(capture))
    reference = artifact_reference(path, stateful.SCHEMA)
    view = derive_stateful_lowering_view(reference,
        stage_capture=artifact_reference(capture_path, capture['schema']))
    assert derive_stateful_lowering_view(path)['stages'] == view['stages']
    assert view['source']['sha256'] == capture['full_source_sha256']
    assert view['identity']['implementation_sha256'] == producer
    assert view['compiler'] == compiler and not view['native_loaded'] and not view['source_F4_admitted']
    expected = {row['index']:row for row in plan['stages'] if row['kind'] != 'COMMIT'}
    assert {row['index'] for row in view['stages']} == set(expected)
    lines = data['source'].decode().splitlines()
    bodies = {row['index']:row['body'] for row in capture['stages']}
    ambiguous = [row for row in view['stages'] if row['kind'] == 'WRITE_SHADOW']
    assert len(ambiguous) == 2 and all(row['mapping_status'] == 'AMBIGUOUS_BODY_MATCHES'
        and len(row['source_line_spans']) == 2 and row['spans_are_candidates'] for row in ambiguous)
    for row in view['stages']:
        assert row['source_action_ids'] == expected[row['index']]['source_action_ids']
        assert row['expr_ids'] == expected[row['index']]['expr_ids']
        for span in row['source_line_spans']:
            assert lines[span['start_line']-1:span['end_line']] == [
                '            '+line for line in bodies[row['index']].split('\n')]
    for role, name in files.items():
        (tmp_path/name).write_bytes(data[role]+b'changed')
        with pytest.raises(ValueError, match='checksum differs'):
            derive_stateful_lowering_view(path)
        (tmp_path/name).write_bytes(data[role])
    capture['stages'][0]['body'] += '\nforeign body'
    capture['stages'][0]['body_sha256'] = sha256(capture['stages'][0]['body'].encode()).hexdigest()
    capture_path.write_text(json.dumps(capture))
    with pytest.raises(ValueError, match='capture bodies/provenance differ'):
        derive_stateful_lowering_view(path, stage_capture=capture_path)
    manifest['plan_sha256'] = 'f'*64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='native core identity differs'):
        derive_stateful_lowering_view(path)

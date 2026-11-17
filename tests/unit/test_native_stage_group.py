"""Full-frame group observers retain original bodies and require supplied ports."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import ctypes
import json
import subprocess

import pytest

from test_cpu_hardware_placement import hardware
from test_stateful_stage_dag import _inputs, _region, _render
from scar.codegen import stateful
from scar.codegen.native_stage_group import normalize_stage_groups, render_native_stage_group
from scar.compiler.model import Expr
from scar.compiler.stateful_model import (document, stateful_digest, StatefulAction,
    StatefulProvider, POWER2_SEMANTICS)


def test_original_frame_preserves_scalar_provider_snapshot_write_and_rejects_missing_dependencies(hardware, tmp_path, monkeypatch):
    region = _region((*_inputs(), Expr('factor', 'neg', ('s',)),
        Expr('powered', 'power2', ('y',), {'provider':'power'}), Expr('rhs', 'mul', ('powered', 'factor'))),
        ('factor', 'powered', 'rhs'))
    region = replace(region, providers=(StatefulProvider('power', 'numpy', 'power', ('np', 'power'), POWER2_SEMANTICS),),
        actions=region.actions+tuple(StatefulAction('write:'+str(i), 'write', 'rhs', 'left', None,
            i, i+1, 'assign', {'port':'w'+str(i)}, {'port':'w'+str(i+1)}) for i in range(2)))
    plan, _, binding, _ = _render(region, hardware)

    def forbidden(*args, **kwargs):
        raise AssertionError('metadata test must not compile, load native or execute a target')

    for owner, name in ((subprocess, 'Popen'), (ctypes, 'CDLL'), (stateful, 'load'),
            (stateful, 'build'), (stateful, '_providers'), (stateful, '_admit')):
        monkeypatch.setattr(owner, name, forbidden)
    path = tmp_path/'artifact.json'

    def emit(actual_plan):
        core = dict(region_sha256=stateful_digest(region), plan_sha256=stateful._digest(actual_plan),
            provider_runtime={'fixture':'NO_PROVIDER_RESOLUTION_OR_EXECUTION'})
        files = dict(region='region.json', plan='plan.json', source='stateful.cpp', library='unloaded.so',
            **{'stateful_abi_v1.h':'stateful_abi_v1.h', 'scar_abi_v1.h':'scar_abi_v1.h'})
        data = dict(region=json.dumps(document(region)).encode(), plan=json.dumps(actual_plan).encode(),
            source=stateful._source(region, actual_plan, binding, core).encode(), library=b'UNLOADED_METADATA_FIXTURE')
        for name in ('stateful_abi_v1.h', 'scar_abi_v1.h'):
            data[name] = (stateful._HERE/name).read_bytes()
        hashes = {role:sha256(raw).hexdigest() for role, raw in data.items()}
        for role, name in files.items(): (tmp_path/name).write_bytes(data[role])
        compiler = dict(argv=['unexecuted-g++'], identity='NOT_QUERIED', options=list(stateful.cpp.COMPILER_OPTIONS))
        build_id = stateful._digest(dict(source=hashes['source'],
            headers={name:hashes[name] for name in ('stateful_abi_v1.h', 'scar_abi_v1.h')},
            compiler=compiler['argv'], identity=compiler['identity'], options=compiler['options'],
            implementation_sha256='0'*64, contract=core))
        manifest = dict(schema=stateful.SCHEMA, schema_version=1, stateful_abi_version=1,
            files=files, sha256=hashes, **core, contract_sha256=stateful._digest(core),
            build_id=build_id, implementation_sha256='0'*64, compiler=compiler,
            cpu_ids=list(binding.cpu_ids), external_assumptions=actual_plan['external_assumptions'],
            **{key:actual_plan[key] for key in ('cpu_threads', 'external_array_bytes', 'workspace_payload_bytes',
                'total_region_array_payload_bytes', 'memory_budget_bytes')})
        path.write_text(json.dumps(manifest))
        return data['source'].decode()

    original = emit(plan)
    source, receipt = render_native_stage_group(path)
    indices = [row['index'] for row in plan['stages'] if row['kind'] != 'COMMIT']
    assert receipt['groups'][0]['stage_indices'] == indices and receipt['cut_count'] == 2
    assert {'SCALAR', 'PROVIDER', 'SNAPSHOT_ROOT', 'WRITE_SHADOW'} <= {row['kind'] for row in receipt['stages']}
    restored = source
    for row in reversed(receipt['insertions']):
        inserted = restored[row['char_start']:row['char_end_exclusive']]
        assert sha256(inserted.encode()).hexdigest() == row['sha256']
        restored = restored[:row['char_start']]+restored[row['char_end_exclusive']:]
    assert restored == original
    assert 'noinline' not in source and 'noipa' not in source
    assert receipt['read_symbol'] in source and receipt['identity_symbol'] in source
    assert receipt['observer_identity'] in source and source != original
    assert receipt['external_ports']['initialization'].startswith('ALL_ABI_PORTS_MUST_BE_EXPLICITLY_SUPPLIED')
    assert not receipt['native_run_entry_exit_clocked'] and not receipt['compiled']
    provider = next(row for row in plan['stages'] if row['kind'] == 'PROVIDER')
    _, partial = render_native_stage_group(path, stage_indices=[provider['index']])
    assert partial['groups'][0]['entry_scalar_ready'] == ['factor']
    assert partial['all_original_stages_execute'] and partial['executed_stage_indices'] == [row['index'] for row in plan['stages']]
    _, multi = render_native_stage_group(path, groups=[[indices[0]], indices[1:]])
    assert [(row['begin_cut'], row['end_cut']) for row in multi['groups']] == [(0, 1), (2, 3)]
    assert normalize_stage_groups(indices) == [indices]
    for bad in ([indices[-1], indices[0]], [indices[0], indices[2]], [999], [True]):
        with pytest.raises(ValueError): render_native_stage_group(path, stage_indices=bad)
    with pytest.raises(ValueError, match='not both'):
        render_native_stage_group(path, indices, groups=[indices])
    broken = deepcopy(plan); broken['stages'][0]['depends_on'] = ['unsupplied:missing-predecessor']
    emit(broken)
    with pytest.raises(ValueError, match='unsupplied or out-of-order dependency'):
        render_native_stage_group(path)

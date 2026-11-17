"""Binding identities/conditions stay separate from independently tested cold parents."""
from copy import deepcopy
from hashlib import sha256
import ctypes
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scar.compiler.model import Expr, SourceRef
from scar.compiler.stateful_model import (StatefulAction, StatefulInput, StatefulRegion,
    REQUIRED_GUARDS, document, stateful_digest)
from scar.compiler.source_summary_binding import (derive_stateful_source_summary_binding,
    verify_stateful_source_summary_binding)
from scar.ir.execution_recipe import artifact_reference
from scar.optimization.resource_solver import semantic_sha256


def test_source_binding_replays_parents_rejects_wrong_identity_and_never_infers_purity(tmp_path, monkeypatch):
    from scar.analysis import build, bounded_entry_calls, interprocedural_summary
    from scar.compiler import native_lowering_view, stateful_candidate

    def forbidden(*args, **kwargs):
        raise AssertionError('binding metadata test must not compile, load native or execute a target')

    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(ctypes, 'CDLL', forbidden)

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return artifact_reference(path, value['schema'])

    entry = tmp_path/'entry.py'; entry.write_text('# metadata fixture, never executed\n')
    entry_sha = sha256(entry.read_bytes()).hexdigest()
    programs = [save(tmp_path/name/'manifest.json', dict(schema='scar.program-build',
        schema_version=1, entry=str(entry), f4=dict(entry_unit='entry-unit'))) for name in ('entry-program', 'native-program')]
    calls = dict(schema='scar.bounded-entry-calls', schema_version=1, program=programs[0],
        producer=artifact_reference(entry, 'scar.python-producer'))
    calls_ref = save(tmp_path/'calls.json', calls)
    original = tmp_path/'method.py'; original.write_text('def update(state):\n    state.arr[:] = state.arr\n')
    source = SourceRef(str(original), sha256(original.read_bytes()).hexdigest(), 'update', 1, 0, 'pycode-v2:'+'a'*64)
    region = StatefulRegion('stateful:binding-fixture', source, ('state',),
        (StatefulInput('arr', ('state', 'arr'), 'array'),),
        (Expr('input', 'input', attrs=dict(name='arr', kind='array', epoch=0)),),
        (StatefulAction('write', 'write', 'input', 'arr', None, 0, 1, 'assign',
            {'port':'before'}, {'port':'after'}),), guards=REQUIRED_GUARDS)
    region_ref = save(tmp_path/'region.json', document(region))
    native_ref = save(tmp_path/'native.json', dict(schema='scar.native-stateful-artifact'))
    plan_ref = save(tmp_path/'plan.json', dict(schema='scar.stateful-stage-plan'))
    candidate_ref = save(tmp_path/'candidate.json', dict(schema='scar.stateful-validation-candidate',
        candidate_id='candidate:fixture', region=region_ref, program=programs[1], native_artifact=native_ref))
    specialization_ref = save(tmp_path/'specialization.json', dict(schema='scar.observed-stateful-specialization',
        candidate=candidate_ref, source_region=region_ref, program=programs[1]))
    body, point, definition, function = ({'body':'original'}, {'point':'original'},
        {'definition':'original'}, {'function':'original'})
    local = dict(function_id='source-function:fixture', body=body, body_point=point,
        definition=definition, source_function=function,
        source=dict(path=str(original), sha256=source.sha256))  # Effects intentionally absent, never NONE.
    summary = dict(schema='scar.interprocedural-source-summary', schema_version=1,
        entry_projection=dict(sha256=semantic_sha256(calls), program=calls['program'], producer=calls['producer']),
        functions=[dict(function_id=local['function_id'], body=body, point=point,
            definition=definition, source_function=function, local=local)])
    summary_path = tmp_path/'summary.json'; save(summary_path, summary)
    requested = dict(path=str(original), sha256=source.sha256, code_key=source.code_key)
    witness = dict(witnessed=True, program=programs[0], requested_source=deepcopy(requested),
        source=artifact_reference(original, 'scar.python-source'), matches=[dict(original_body=body,
            original_body_point=point, original_function_definition=definition,
            conditions=['ORIGINAL_UNDISCHARGED_CALL_CONDITION'], remaining_fields=['UNKNOWN_RECEIVER'])])
    invoked = []

    class Session:
        def __init__(self, directory):
            report = json.loads((Path(directory)/'manifest.json').read_bytes())
            self._f4 = SimpleNamespace(path=Path(directory), report=report,
                catalog=SimpleNamespace(fact_store=SimpleNamespace(model_units={
                    'entry-unit':dict(path=str(entry), raw_sha256=entry_sha)})))
        def __enter__(self): return self
        def __exit__(self, *args): pass

    def admission(reference):
        invoked.append(('native', reference)); return dict(native_metadata_verified=True)

    def verify(session, report):
        invoked.append(('summary', report['schema'])); return deepcopy(report)

    def source_witness(session, report, **kwargs):
        invoked.append(('witness', kwargs)); return deepcopy(witness)

    monkeypatch.setattr(build, 'BuildSession', Session)
    monkeypatch.setattr(stateful_candidate, 'check_observed_specialization', admission)
    monkeypatch.setattr(interprocedural_summary, 'verify_interprocedural_summary', verify)
    monkeypatch.setattr(bounded_entry_calls, 'derive_entry_source_witness', source_witness)
    monkeypatch.setattr(native_lowering_view, 'derive_stateful_lowering_view', lambda ref:
        dict(native_manifest=ref, identity=dict(build_id='fixture', region_sha256=stateful_digest(region)), plan=plan_ref))
    result = derive_stateful_source_summary_binding(summary_path, specialization_ref, entry_calls=calls_ref)
    assert [row[0] for row in invoked] == ['native', 'summary', 'witness']
    assert result['native_stateful_write_actions'][0]['target_path'] == ['state', 'arr']
    assert result['entry_witness']['matches'][0]['conditions'] == witness['matches'][0]['conditions']
    assert result['missing_effects_interpretation'] == 'UNKNOWN_NOT_NONE_OR_PURE'
    assert not result['purity_proven'] and not result['native_legality_derived_from_summary']
    changed = deepcopy(result); changed['effects_closed'] = True
    with pytest.raises(ValueError, match='differs from cold replay'):
        verify_stateful_source_summary_binding(changed, summary_path, specialization_ref, entry_calls=calls_ref)
    for mutate in ('projection', 'point', 'source'):
        changed = deepcopy(summary)
        if mutate == 'projection': changed['entry_projection']['sha256'] = 'f'*64
        elif mutate == 'point': changed['functions'][0]['point'] = {'point':'other-body'}
        else: changed['functions'][0]['local']['source']['sha256'] = 'f'*64
        save(summary_path, changed)
        with pytest.raises(ValueError):
            derive_stateful_source_summary_binding(summary_path, specialization_ref, entry_calls=calls_ref)
    save(summary_path, summary)
    witness['requested_source']['code_key'] = 'pycode-v2:'+'b'*64
    with pytest.raises(ValueError, match='native source identity'):
        derive_stateful_source_summary_binding(summary_path, specialization_ref, entry_calls=calls_ref)
    witness['requested_source'] = requested
    entry.write_text('# changed entry version\n')
    with pytest.raises(ValueError, match='entry source version'):
        derive_stateful_source_summary_binding(summary_path, specialization_ref, entry_calls=calls_ref)

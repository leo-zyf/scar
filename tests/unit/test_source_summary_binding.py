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
    from contextlib import contextmanager
    from concurrent.futures import ThreadPoolExecutor
    import pickle
    import gzip
    from scar.analysis import build, bounded_entry_calls, interprocedural_summary
    from scar.analysis import source_fact_materials, source_summary_storage
    from scar.codegen import stateful
    from scar.compiler.admission_context import AdmissionContext, require_admission_session
    from scar.compiler.source_summary_reader_versions import _BINDING_PREDECESSOR
    from scar.analysis.replay_transaction import replay_transaction
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
    # The logical collection shard deliberately does not exist. Only the
    # official unit_parent route reaches this primitive compressed table.
    unit='fixture-unit';primitive=tmp_path/'primitive';collection=tmp_path/'collection';catalog_path=tmp_path/'catalog'
    physical=primitive/'units'/'facts.json.gz';physical.parent.mkdir(parents=True)
    payload=(json.dumps(dict(unit=unit,value=7))+'\n').encode();compressed=gzip.compress(payload)
    physical.write_bytes(compressed)
    primitive_ref=save(primitive/'result.json',dict(schema='fixture.primitive-facts'))
    collection_ref=save(collection/'result.json',dict(schema='fixture.collection-facts'))
    row=dict(facts='units/facts.json.gz',sha256=sha256(compressed).hexdigest(),bytes=len(compressed),
        decoded_bytes=len(payload),decoded_sha256=sha256(payload).hexdigest(),records=1,facts_format='COMPILE_CONTEXT_F4_JSONL')
    parent=SimpleNamespace(path=primitive,raw=(primitive/'result.json').read_bytes(),
        report=dict(schema=primitive_ref['schema']),parents=(),rows={unit:row})
    selected=SimpleNamespace(path=collection,raw=(collection/'result.json').read_bytes(),
        report=dict(schema=collection_ref['schema']),parents=[parent],unit_parent={unit:0},rows={unit:row})
    catalog_report=dict(schema='fixture.source-catalog',context_facts=dict(path=str(collection/'result.json'),
        sha256=collection_ref['sha256'],bytes=collection_ref['size_bytes']))
    save(catalog_path/'result.json',catalog_report)
    store=SimpleNamespace(rows={unit:row},report=selected.report,validate_parents=lambda:None,
        source_parent_for=lambda key:selected,model_units={'entry-unit':dict(path=str(entry),raw_sha256=entry_sha)})
    catalog=SimpleNamespace(path=catalog_path,report=catalog_report,fact_store=store)
    logical=dict(uri=str(collection/row['facts']),sha256=row['sha256'],size_bytes=row['bytes'],schema='scar.source-facts-unit')
    fact=dict(artifact=logical,locator='#/records/0/value')
    assert not Path(logical['uri']).exists()
    monkeypatch.setattr(source_fact_materials,'read_source_catalog',lambda value:value)
    original = tmp_path/'method.py'; original.write_text('def update(state):\n    state.arr[:] = state.arr\n')
    source = SourceRef(str(original), sha256(original.read_bytes()).hexdigest(), 'update', 1, 0, 'pycode-v2:'+'a'*64)
    region = StatefulRegion('stateful:binding-fixture', source, ('state',),
        (StatefulInput('arr', ('state', 'arr'), 'array'),),
        (Expr('input', 'input', attrs=dict(name='arr', kind='array', epoch=0)),),
        (StatefulAction('write', 'write', 'input', 'arr', None, 0, 1, 'assign',
            {'port':'before'}, {'port':'after'}),), guards=REQUIRED_GUARDS)
    region_ref = save(tmp_path/'region.json', document(region))
    provider = tmp_path/'provider.bin';provider.write_bytes(b'UNLOADED_METADATA_PROVIDER')
    plan = dict(schema='scar.stateful-stage-plan', cpu_threads=1,resource_binding=dict(cpu_ids=[0],evidence=[]))
    plan_ref = save(tmp_path/'plan.json', plan)
    native_ref = save(tmp_path/'native.json', dict(schema='scar.native-stateful-artifact',
        files=dict(region='region.json',plan='plan.json'),cpu_threads=1,
        implementation_sha256=stateful.implementation_sha256(),compiler=dict(options=list(stateful.cpp.COMPILER_OPTIONS)),
        provider_runtime=dict(loaded_binary=dict(path=str(provider),sha256=sha256(provider.read_bytes()).hexdigest(),bytes=provider.stat().st_size))))
    domain_ref=save(tmp_path/'domain.json',dict(schema='scar.stateful-cpu-domain'))
    candidate_ref = save(tmp_path/'candidate.json', dict(schema='scar.stateful-validation-candidate',
        candidate_id='candidate:fixture', region=region_ref, program=programs[1], native_artifact=native_ref))
    specialization_ref = save(tmp_path/'specialization.json', dict(schema='scar.observed-stateful-specialization',
        candidate=candidate_ref, source_region=region_ref, program=programs[1],domain=domain_ref))
    body, point, definition, function = ({'body':'original'}, {'point':'original'},
        {'definition':'original'}, {'function':'original'})
    local = dict(function_id='source-function:fixture', body=body, body_point=point,
        definition=definition, source_function=function,
        source=dict(path=str(original), sha256=source.sha256),source_refs=[logical,fact])  # Effects absent, never NONE.
    summary = dict(schema=interprocedural_summary.SCHEMA, schema_version=1,
        entry_projection=dict(sha256=semantic_sha256(calls), program=calls['program'], producer=calls['producer']),
        functions=[dict(function_id=local['function_id'], body=body, point=point,
            definition=definition, source_function=function, local=local)])
    summary_path = tmp_path/'summary.json'; save(summary_path, summary)
    requested = dict(path=str(original), sha256=source.sha256, code_key=source.code_key)
    witness = dict(witnessed=True, program=programs[0], requested_source=deepcopy(requested),
        source=artifact_reference(original, 'scar.python-source'), matches=[dict(original_body=body,
            original_body_point=point, original_function_definition=definition,
            conditions=['ORIGINAL_UNDISCHARGED_CALL_CONDITION'], remaining_fields=['UNKNOWN_RECEIVER'])])
    invoked = [];created=[];closed=[]

    class Session:
        def __init__(self, directory):
            created.append(str(directory))
            report = json.loads((Path(directory)/'manifest.json').read_bytes())
            self._f4 = SimpleNamespace(path=Path(directory), report=report,closed=False,catalog=catalog)
        def __enter__(self): return self
        def close(self):self._f4.closed=True;closed.append(str(self._f4.path))
        def __exit__(self, *args):self.close()

    def admission(reference,*,_session=None):
        assert require_admission_session(_session,programs[1]) is _session
        invoked.append(('native', reference)); return dict(native_metadata_verified=True)

    @contextmanager
    def verify(session, report, *, entry_calls):
        token=object();session._test_entry_token=token
        invoked.append(('summary', report['schema']));yield deepcopy(report),token
        del session._test_entry_token

    def source_witness(session, admission, **kwargs):
        assert admission is session._test_entry_token
        invoked.append(('witness', kwargs)); return deepcopy(witness)

    def guard(self,session,reference):
        assert not session._f4.closed
        assert artifact_reference(session._f4.path/'manifest.json','scar.program-build')==reference
        self.watch(reference)

    monkeypatch.setattr(build, 'BuildSession', Session)
    monkeypatch.setattr(stateful_candidate, 'check_observed_specialization', admission)
    monkeypatch.setattr(AdmissionContext,'_guard_session',guard)  # F4 store semantics are tested by their original readers.
    monkeypatch.setattr(interprocedural_summary, '_verified_interprocedural_summary', verify)
    monkeypatch.setattr(bounded_entry_calls, '_derive_entry_source_witness_admitted', source_witness)
    monkeypatch.setattr(stateful,'_admit',lambda *args:(region,deepcopy(plan),SimpleNamespace(cpu_ids=(0,))))
    monkeypatch.setattr(native_lowering_view, 'derive_stateful_lowering_view', lambda ref:
        dict(native_manifest=ref, identity=dict(build_id='fixture', region_sha256=stateful_digest(region)),
            plan=plan_ref,region=region_ref,source=artifact_reference(original,'scar.cpp-source'),
            library=artifact_reference(provider,'scar.native-library'),headers={}))
    with AdmissionContext() as context:
        first=context.stateful(specialization_ref)
        first['region']['guards']=[]
        assert context.stateful(specialization_ref)['region']['guards']
        result = derive_stateful_source_summary_binding(summary_path, specialization_ref, entry_calls=calls_ref,_context=context)
        assert len(created)==2 and len(invoked)==3
        assert str(physical) in context._watched and logical['uri'] not in context._watched
        context._watch_tree([logical,fact],_program=programs[0])
        with pytest.raises(ValueError,match='outside the recorded catalog'):
            context._watch_tree([dict(logical,uri=str(collection/'unknown.json.gz'))],_program=programs[0])
        assert local['source_refs'][0]['uri']==logical['uri']
        assert context.program(programs[0]) is context.program(programs[0])
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(ValueError):pool.submit(context.program,programs[0]).result()
        with pytest.raises(TypeError):pickle.dumps(context)
        with pytest.raises(TypeError):deepcopy(context)
        with replay_transaction():
            with pytest.raises(ValueError,match='another metadata replay scope'):context.program(programs[0])
        with pytest.raises(ValueError):require_admission_session(context.program(programs[0]),programs[1])
        sharded = dict(summary, calls=[], tokens=[], sccs=[], dependency_sets={}, unknown_facts={})
        sharded['id'] = 'interprocedural-summary:' + semantic_sha256(sharded)
        with monkeypatch.context() as small:
            small.setattr(source_summary_storage, '_LIMIT', 0)
            sharded_ref = source_summary_storage.write_source_summary(tmp_path/'sharded.json', sharded)
        tabular = derive_stateful_source_summary_binding(sharded_ref, specialization_ref,
            entry_calls=calls_ref, _context=context)
        assert tabular['bindings'][0]['local_summary'] == dict(artifact=sharded_ref, locator='#/functions/0/local')
        assert tabular['bindings'][0]['local_summary_sha256'] == semantic_sha256(local)
        folder = tmp_path/'sharded.json.tables'
        assert len(tuple(folder.iterdir())) == 6
        assert all(str(path) in context._watched for path in folder.iterdir())
    assert len(closed)==2
    with pytest.raises(ValueError):context.stateful(specialization_ref)
    assert [row[0] for row in invoked] == ['native', 'summary', 'witness', 'summary', 'witness']
    with pytest.raises(ValueError, match='admission dependency'):
        with AdmissionContext() as context:
            source_summary_storage.read_source_summary(sharded_ref, watch=context.watch)
            (folder/'functions-000000.jsonl.gz').write_bytes(b'changed physical summary')
            context.validate()
    assert result['native_stateful_write_actions'][0]['target_path'] == ['state', 'arr']
    assert result['entry_witness']['matches'][0]['conditions'] == witness['matches'][0]['conditions']
    assert result['missing_effects_interpretation'] == 'UNKNOWN_NOT_NONE_OR_PURE'
    assert not result['purity_proven'] and not result['native_legality_derived_from_summary']
    before=len(invoked)
    with AdmissionContext() as cold:
        cold.stateful(specialization_ref)
    assert len(invoked)==before+1  # An independent context performs a real factory call again.
    with pytest.raises(ValueError,match='admission dependency'):
        with AdmissionContext() as fact_context:
            fact_context.program(programs[0])
            fact_context._watch_tree([logical,fact],_program=programs[0])
            physical.write_bytes(b'changed physical facts')
            fact_context.validate()
    physical.write_bytes(compressed)
    with pytest.raises(ValueError,match='version changed'):
        with AdmissionContext() as changed_context:
            changed_context.stateful(specialization_ref)
            provider.write_bytes(b'changed');provider.write_bytes(b'UNLOADED_METADATA_PROVIDER')
            changed_context.stateful(specialization_ref)
    changed = deepcopy(result); changed['effects_closed'] = True
    with pytest.raises(ValueError, match='differs from cold replay'):
        verify_stateful_source_summary_binding(changed, summary_path, specialization_ref, entry_calls=calls_ref)
    previous_sha, previous_size = _BINDING_PREDECESSOR
    preceding = dict(result, producer_source=dict(result['producer_source'],
        sha256=previous_sha, size_bytes=previous_size))
    assert verify_stateful_source_summary_binding(preceding, summary_path, specialization_ref,
        entry_calls=calls_ref) == preceding
    preceding['producer_source']['sha256'] = '0'*64
    with pytest.raises(ValueError, match='binding producer differs'):
        verify_stateful_source_summary_binding(preceding, summary_path, specialization_ref, entry_calls=calls_ref)
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

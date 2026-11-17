"""One live replay capability reuses entry work and rejects escaped authority."""
from contextvars import Context, copy_context
from copy import deepcopy
from hashlib import sha256
import gzip
import json
from pathlib import Path
from threading import Thread
from types import MappingProxyType, SimpleNamespace as NS

import pytest

from scar.analysis import bounded_entry_calls as B, interprocedural_summary as I
from scar.analysis.build import BuildSession
from scar.analysis.source_declared_catalog import DeclaredSourceCatalog, SCHEMA as CATALOG_SCHEMA
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import encode


def test_live_entry_admission_owns_one_replay_and_rechecks_material(tmp_path, monkeypatch):
    directory = tmp_path/'program'; directory.mkdir()
    manifest = directory/'manifest.json'
    report = dict(schema='scar.program-build', backend='F4', rules={})
    manifest.write_text(json.dumps(report))
    model_leaf = dict(state='UNKNOWN', parameters=('first', 'second'))
    models = MappingProxyType({'unit':MappingProxyType({
        'artifacts':MappingProxyType({'facts':MappingProxyType(model_leaf)})})})
    # Original collection metadata keeps a logical path under the collection,
    # while the immutable gzip shard remains only in its primitive parent.
    collection = tmp_path/'collection'; collection.mkdir()
    primitive = tmp_path/'primitive'; primitive.mkdir()
    physical = primitive/'units/unit/facts.json.gz'; physical.parent.mkdir(parents=True)
    decoded = json.dumps(dict(unit='unit', functions=[])).encode()
    physical.write_bytes(gzip.compress(decoded, mtime=0))
    row = dict(unit='unit', facts='units/unit/facts.json.gz',
        sha256=sha256(physical.read_bytes()).hexdigest(), bytes=physical.stat().st_size,
        decoded_sha256=sha256(decoded).hexdigest(), decoded_bytes=len(decoded))
    primitive_raw = json.dumps(dict(schema='scar.normal-source-link-facts')).encode()
    (primitive/'result.json').write_bytes(primitive_raw)
    parent = NS(path=primitive, raw=primitive_raw, parents=(), rows={'unit':row},
        report={'schema':'scar.normal-source-link-facts'})
    collection_raw = json.dumps(dict(schema='scar.normal-source-fact-collection')).encode()
    (collection/'result.json').write_bytes(collection_raw)
    selected = NS(path=collection, raw=collection_raw, parents=(parent,), unit_parent={'unit':0},
        rows={'unit':row}, report={'schema':'scar.normal-source-fact-collection'})
    catalog_path = tmp_path/'catalog'; catalog_path.mkdir()
    catalog_report = dict(schema=CATALOG_SCHEMA, facts=dict(path=str(collection/'result.json'),
        sha256=sha256(collection_raw).hexdigest(), bytes=len(collection_raw)))
    (catalog_path/'result.json').write_text(json.dumps(catalog_report))
    catalog = object.__new__(DeclaredSourceCatalog)
    catalog.path, catalog.report = catalog_path, catalog_report
    catalog.fact_store = NS(model_units=models, rows={'unit':row}, report=selected.report,
        source_parent_for=lambda unit:selected, validate_parents=lambda:None)
    catalog.interface_store = NS(validate_parents=lambda:None)
    owner = NS(path=directory, report=report, closed=False, catalog=catalog,
        view=NS(units={}), providers={}, _admitted={})
    session = object.__new__(BuildSession); session._f4 = owner
    other = object.__new__(BuildSession); other._f4 = owner
    entry = dict(schema=B.SCHEMA, schema_version=1,
        program=B._artifact(manifest, 'scar.program-build'),
        producer=B._artifact(B.__file__, 'scar.python-producer'), summary={},
        source_bodies=[], observed_targets=[], frontiers=[], observed_target_frontiers=[], interfaces=[])
    counts = [0]
    def derive(actual):
        assert actual is session
        counts[0] += 1
        return deepcopy(entry)
    monkeypatch.setattr(B, 'derive_bounded_entry_calls', derive)
    with B._admit_entry_calls(session, entry) as admitted:
        summary = I._derive_interprocedural_summary_admitted(session, admitted)
    counts[0] = 0
    path = tmp_path/'source.py'; path.write_text('metadata only; never compiled\n')
    logical = ProviderArtifact(str(collection/row['facts']), row['sha256'], row['bytes'], 'scar.source-facts-unit')
    with I._verified_interprocedural_summary(session, summary, entry_calls=entry) as (checked, admitted):
        assert checked == summary and counts[0] == 1
        witness = B._derive_entry_source_witness_admitted(session, admitted, path=path,
            sha256=sha256(path.read_bytes()).hexdigest(), code_key='unobserved-code-id')
        assert not witness['witnessed'] and counts[0] == 1
        B._entry_admission_dependencies(session, admitted, (encode(logical),))
        assert not Path(logical.uri).exists() and str(physical) in admitted._versions
        detached = B._entry_projection(session, admitted); detached['schema'] = 'tampered'
        assert B._entry_projection(session, admitted)['schema'] == B.SCHEMA
        with pytest.raises(ValueError, match='owner'):
            B._entry_projection(other, admitted)
        with pytest.raises(ValueError, match='context'):
            Context().run(B._entry_projection, session, admitted)
        errors = []
        def foreign():
            try:B._entry_projection(session, admitted)
            except ValueError as error:errors.append(str(error))
        thread = Thread(target=copy_context().run, args=(foreign,))
        thread.start(); thread.join()
        assert errors and 'thread' in errors[0]
        with pytest.raises(TypeError, match='copied'):
            deepcopy(admitted)
        with pytest.raises(ValueError, match='owner'):
            B._entry_projection(session, {'verified':True})
    assert counts[0] == 1
    with pytest.raises(ValueError, match='owner'):
        B._entry_projection(session, admitted)
    with pytest.raises(TypeError, match='factory'):
        B.EntryAdmission()
    unknown = ProviderArtifact(str(collection/'unknown.json.gz'), row['sha256'], row['bytes'], logical.schema)
    with pytest.raises(ValueError, match='outside the recorded catalog'):
        B._resolve_fact_artifact(catalog, encode(unknown))
    changed = deepcopy(entry); changed['verified'] = True
    with pytest.raises(ValueError, match='differs'):
        with B._admit_entry_calls(session, changed):pass
    original = manifest.read_bytes()
    with pytest.raises(ValueError, match='ProgramBuild manifest changed'):
        with B._admit_entry_calls(session, entry) as admitted:
            manifest.write_text('{}')
            B._entry_projection(session, admitted)
    manifest.write_bytes(original)
    model_leaf['unsupported'] = object()
    with pytest.raises(TypeError, match='not JSON serializable'):
        with B._admit_entry_calls(session, entry):pass
    assert not session._scar_entry_admissions
    del model_leaf['unsupported']
    with pytest.raises(ValueError, match='parent metadata changed in memory'):
        with B._admit_entry_calls(session, entry) as admitted:
            model_leaf['state'] = 'NONE'
            B._entry_projection(session, admitted)
    model_leaf['state'] = 'UNKNOWN'
    stored = physical.read_bytes()
    with pytest.raises(ValueError, match='dependency version changed'):
        with B._admit_entry_calls(session, entry) as admitted:
            B._entry_admission_dependencies(session, admitted, (encode(logical),))
            physical.write_bytes(b'tampered')
            B._entry_projection(session, admitted)
    physical.write_bytes(stored)
    with pytest.raises(ValueError, match='live owner'):
        with B._admit_entry_calls(session, entry) as admitted:
            owner.closed = True
            B._entry_projection(session, admitted)

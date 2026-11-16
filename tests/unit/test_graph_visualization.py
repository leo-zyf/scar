"""Bounded presentation retains qualified IR; it never executes the workload."""
from copy import deepcopy
from dataclasses import dataclass, replace
from html.parser import HTMLParser
import json

import pytest

from scar.analysis.program_view import ProgramView
from scar.ir.program import view as V
from scar.ir.program.dependencies import (DependencyCondition, DependencyEdge,
    DependencyEvidence, DependencyFrame, DependencyFrontier, DependencyKind as K,
    DependencyMeaning as M, DependencyNeighborhood, DependencyPoint as P,
    DependencyStrength as S, GraphNamespace)
from scar.ir.program.ids import OperationID, RegionID, ValueID
from scar.ir.program.providers import ProviderArtifact
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.record_codec import encode
from scar.visualization.project import project_view


ATTACK = '</script><script>globalThis.viewerInjected=true</script>'


@dataclass(frozen=True)
class _Record:
    id: object
    kind: str
    description: str = ATTACK
    trip_count: int | None = None


def _program_view():
    artifact = ProviderArtifact('/evidence/result.json', '0' * 64, 1, 'test.evidence')
    locator = SourceArtifactLocator(artifact, '0' * 64, 1, 'canonical-json')
    outer = GraphNamespace(locator, 'test.outer.v1', RegionID('outer'))
    callee = GraphNamespace(locator, 'test.shared.v1', RegionID('shared'))
    root = P(outer, outer.root, None)
    a, b = (P(outer, OperationID(name), None) for name in ('call-a', 'call-b'))
    value = P(outer, ValueID('input'), None)
    loop = P(outer, RegionID('loop'), None)
    iteration = P(outer, OperationID('shared-loop-body'), None)
    # Two original static interface selections share a body but not a context.
    ca, cb = (P(callee, callee.root, (DependencyFrame(outer, p.node,
        None, None, callee, entry=callee.root),)) for p in (a, b))
    points = (root, a, b, value, loop, iteration, ca, cb)
    records = {p: _Record(p.node, 'LOOP' if p == loop else 'ORIGINAL',
        trip_count=1_000_000 if p == loop else None) for p in points}
    children = {root: (a, b, value, loop), a: (ca,), b: (cb,), loop: (iteration,)}
    evidence = (DependencyEvidence('test.original.v1', (locator,), (root,)),)
    condition = DependencyCondition('test.source', 'SELECTED_TARGET', a)
    edges = (
        DependencyEdge(value, a, K.OPERAND, M.VALUE, evidence),
        DependencyEdge(a, ca, K.CALL_TARGET, M.BINDING, evidence, (condition,), S.MAY),
        DependencyEdge(b, cb, K.CALL_TARGET, M.BINDING, evidence, (condition,), S.MAY),
        DependencyEdge(root, loop, K.CONTAINS, M.STRUCTURE, evidence),
    )
    class Evidence:
        owned_namespaces = (outer, callee)
        def describe(self, p):
            if p not in records:
                return None
            kind = V.ProgramEntityKind.REGION if type(p.node) is RegionID else (
                V.ProgramEntityKind.OPERATION if type(p.node) is OperationID else V.ProgramEntityKind.VALUE)
            role = {V.ProgramEntityKind.REGION: V.ProgramNodeRole.REGION,
                V.ProgramEntityKind.OPERATION: V.ProgramNodeRole.OPERATION,
                V.ProgramEntityKind.VALUE: V.ProgramNodeRole.CORE_VALUE}[kind]
            return V.ProgramNode(p, kind, role, None if p == root else root, None)
        def original(self, p):
            return records[p]
        def expand(self, p):
            return children.get(p, ())
        def explain(self, p):
            raise AssertionError('viewer must not derive effects or execute source')
        def incoming(self, p):
            return DependencyNeighborhood(tuple(e for e in edges if e.target == p))
        def outgoing(self, p):
            return DependencyNeighborhood(tuple(e for e in edges if e.source == p),
                (DependencyFrontier(p, 'BODY_NOT_EXECUTED', 'static selection only'),) if p == a else ())
    view = ProgramView(None)
    view.register(Evidence())
    return view, root, (ca, cb), records, edges


def test_program_projection_preserves_context_evidence_records_and_bounded_frontiers(monkeypatch):
    import subprocess
    def no_child(*args, **kwargs):
        raise AssertionError('viewer must not execute a workload or external tool')
    monkeypatch.setattr(subprocess, 'Popen', no_child)
    view, root, callees, records, originals = _program_view()
    result = project_view(view, (root,), max_nodes_per_part=20,
        max_edges_per_part=20, max_parts=8, max_depth=2)
    assert result['coverage']['target_program_executed'] is False
    assert result['coverage']['execution_proof'] is False
    assert not result['coverage']['loops_unrolled']
    assert len(result['parts']) == 4  # one loop body and two qualified shared callees
    root_part = next(p for p in result['parts'] if p['id'] == result['root'])
    nodes = root_part['nodes']
    callee_nodes = [n for n in nodes if n['point'] in [encode(p) for p in callees]]
    assert len(callee_nodes) == 2 and len({n['id'] for n in callee_nodes}) == 2
    for node in nodes:
        assert node['details']['record'] in [encode(r) for r in records.values()]
        assert node['details']['record']['description'] == ATTACK
        assert node['source']['locator']['artifact']['uri'] == '/evidence/result.json'
    assert any(n['details']['record']['trip_count'] == 1_000_000 for n in nodes)
    may = [e for e in root_part['edges'] if e['strength'] == 'MAY']
    assert len(may) == 2 and all(e['conditions'] and e['evidence'] for e in may)
    assert any(f['kind'] == 'BODY_NOT_EXECUTED' for f in root_part['frontiers'])
    for part in result['parts']:
        ids = {n['id'] for n in part['nodes']}
        assert all(e['source'] in ids and e['target'] in ids for e in part['edges'])
        assert all(e['runtime_occurrence'] is False for e in part['edges'])
        assert {e['kind'] for e in part['edges']} <= {e.kind.value for e in originals}
    bounded = project_view(view, (root,), max_nodes_per_part=2,
        max_edges_per_part=1, max_parts=1, max_depth=0)
    assert len(bounded['parts']) == 1
    part = bounded['parts'][0]
    assert len(part['nodes']) <= 2 and len(part['edges']) <= 1
    assert part['boundaries'] and not bounded['coverage']['complete']
    assert all(e['source'] in {n['id'] for n in part['nodes']} and
        e['target'] in {n['id'] for n in part['nodes']} for e in part['edges'])


def test_schedule_bundle_is_offline_safe_and_keeps_original_ids_without_ir_changes(tmp_path, monkeypatch):
    import subprocess
    from test_versioned_schedule import ordered_state_schedule
    from scar.optimization.schedule import schedule_document
    from scar.visualization.adapters import project_schedule
    from scar.visualization.bundle import write_view_bundle

    def no_child(*args, **kwargs):
        raise AssertionError('viewer export must not execute a workload or external tool')
    monkeypatch.setattr(subprocess, 'Popen', no_child)
    schedule = replace(ordered_state_schedule(), region_id=ATTACK + '\u2028\u2029&')
    before = schedule_document(schedule)
    projection = project_schedule(schedule)
    projection['parts'][0]['nodes'][0]['details']['__proto__'] = {'original': ATTACK}
    untouched = deepcopy(projection)
    directory = tmp_path / 'offline'
    report = write_view_bundle(projection, directory, title=ATTACK)
    assert schedule_document(schedule) == before and projection == untouched
    assert report['target_program_executed'] is False and report['ir_modified'] is False
    assert report['network_required_at_view_time'] is False
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['title'] == ATTACK
    manifest_js = (directory / 'manifest.js').read_text()
    manifest_payload = manifest_js.removeprefix('window.SCAR_GRAPH_MANIFEST=JSON.parse(').removesuffix(');\n')
    assert json.loads(json.loads(manifest_payload)) == manifest
    assert ATTACK not in manifest_js and '\\u003c' in manifest_js
    parts = []
    for spec in manifest['parts'].values():
        file = directory / spec['file']
        assert file.resolve().is_relative_to(directory.resolve() / 'parts')
        script = file.read_text()
        assert '</script>' not in script and '\u2028' not in script and '\u2029' not in script
        part = json.loads(json.loads(script.removeprefix('window.SCAR_VIEW.register(JSON.parse(').removesuffix('));\n')))
        parts.append(part)
        ids = {n['id'] for n in part['nodes']}
        assert all(e['source'] in ids and e['target'] in ids for e in part['edges'])
        assert all('children_parts' in n for n in part['nodes'])
        assert all(child in manifest['parts'] for n in part['nodes'] for child in n['children_parts'])
    task_records = {n['details']['original_task_id']: n['details']['record']
        for p in parts for n in p['nodes'] if 'original_task_id' in n['details']}
    assert task_records == {t.id: encode(t) for t in schedule.tasks}
    assert parts[0]['nodes'][0]['details']['__proto__'] == {'original': ATTACK}
    assert any(n['label'] == schedule.region_id for p in parts for n in p['nodes'])
    class Scripts(HTMLParser):
        sources = []
        def handle_starttag(self, tag, attrs):
            if tag == 'script':
                self.sources.append(dict(attrs).get('src'))
    parser = Scripts()
    html = (directory / 'index.html').read_text()
    parser.feed(html)
    assert ATTACK not in html
    assert parser.sources == ['vendor/cytoscape.min.js', 'manifest.js', 'viewer.js']
    assert all((directory / source).is_file() for source in parser.sources)
    assert 'innerHTML' not in (directory / 'viewer.js').read_text()
    with pytest.raises(ValueError, match='fresh viewer output'):
        write_view_bundle(projection, directory)
    broken = deepcopy(projection)
    broken['parts'][0]['edges'][0]['target'] = 'missing-original-endpoint'
    with pytest.raises(ValueError, match='both original endpoints'):
        write_view_bundle(broken, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists()

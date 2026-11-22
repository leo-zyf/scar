"""One finite loop/branch/child packet preserves the original edge algebra."""
from dataclasses import replace
from types import MappingProxyType

import pytest

from scar.analysis import normal_graph
from scar.analysis.source_productions import _digest
from scar.analysis.source_region_projection import RegionProjectionIndex
from scar.ir.program import normal_graph as G, source_productions as P
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.ids import LocationID
from scar.ir.program.model import ValueKind, ValueType
from scar.ir.program.source_contexts import ContextState, OwnerRef
from scar.ir.program.source_declarations import SourceMaterialDomain, SourceTemplateKey, SourceTemplateLink, SourceTemplateNode, SourceTemplateOrigin
from scar.ir.record_codec import encode


def test_region_packets_emit_only_demanded_body_and_merge_to_original_edges(monkeypatch):
    reference = ValueType(ValueKind.REFERENCE)
    boolean = ValueType(ValueKind.DATA, 'builtins.bool')
    world = ValueType(ValueKind.STATE, location=LocationID('world'))
    source = P.ProductionOccurrence(0)
    key, child_key = (SourceTemplateKey(i, P.ProductionPurpose.VALUE) for i in (0, 1))
    context = ContextState(OwnerRef(0, 'ROOT'), 0, 0, 0)
    rid, sid = P.ProductionRegionID, P.ProductionStepID
    parameter = lambda name, i: P.ProductionParameter(rid(name), i)
    result = lambda name, i: P.ProductionResult(sid(name), i)
    completion = lambda *types: P.ProductionCompletion(C.FALLTHROUGH, types)
    regions, steps = [], []

    def region(name, parent, kind, types, names):
        regions.append(P.ProductionRegion(rid(name), None if parent is None else rid(parent),
            kind, source, context, (reference, world), (completion(*types),), tuple(sid(n) for n in names)))

    def step(name, owner, instruction, arguments, types):
        steps.append(P.ProductionStep(sid(name), rid(owner), source, instruction,
            arguments, (completion(*types),)))

    region('root', None, P.ProductionRegionKind.EXECUTION, (world,), ('child', 'declare', 'start', 'done'))
    region('loop', 'root', P.ProductionRegionKind.LOOP, (world,), ('drive',))
    region('condition', 'loop', P.ProductionRegionKind.LOOP_CONDITION, (boolean, world), ('test', 'tested'))
    region('body', 'loop', P.ProductionRegionKind.LOOP_BODY, (world,), ('choose', 'branch', 'body_done'))
    region('yes', 'body', P.ProductionRegionKind.ALTERNATIVE, (world,), ('yes_done',))
    region('no', 'body', P.ProductionRegionKind.ALTERNATIVE, (world,), ('no_done',))
    region('deferred', 'root', P.ProductionRegionKind.DEFERRED, (world,), ('deferred_done',))
    step('child', 'root', P.EvaluateChild(P.SlotRef(0, 'FIELD', 0), P.ProductionPurpose.VALUE),
        (parameter('root', 0), parameter('root', 1)), (world,))
    environment = P.DeferredEnvironment(1, 0, (), (), P.DeferredBodyKind.FUNCTION,
        world, P.ExceptionABI.EXCEPTION_WORLD)
    step('declare', 'root', P.DeferRegion(rid('deferred'), environment),
        (result('child', 0),), (reference, world))
    step('start', 'root', P.ApplyRegion(P.ProductionRegionCall(rid('loop'),
        (parameter('root', 0), result('declare', 1)))), (), (world,))
    step('done', 'root', P.TransferCompletion(C.FALLTHROUGH), (result('start', 0),), (world,))
    step('drive', 'loop', P.LoopControl(rid('condition'), rid('body'), invariant_count=1),
        (parameter('loop', 0), parameter('loop', 1)), (world,))
    for name, owner in (('test', 'condition'), ('choose', 'body')):
        step(name, owner, P.ApplyLanguageAction(P.LanguageActionKind.TRUTH),
            (result('declare', 0) if name == 'choose' else parameter(owner, 0),
                parameter(owner, 1)), (boolean, world))
    step('tested', 'condition', P.TransferCompletion(C.FALLTHROUGH),
        (result('test', 0), result('test', 1)), (boolean, world))
    step('branch', 'body', P.BranchControl(*(
        P.ProductionRegionCall(rid(name), (parameter('root', 0), result('choose', 1)))
        for name in ('yes', 'no'))), (result('choose', 0),), (world,))
    step('body_done', 'body', P.TransferCompletion(C.FALLTHROUGH), (result('branch', 0),), (world,))
    for name in ('yes', 'no', 'deferred'):
        step(name+'_done', name, P.TransferCompletion(C.FALLTHROUGH), (parameter(name, 1),), (world,))
    basis = P.ProductionInput('/source/packet.py', '/source', *('a'*64 for _ in range(8)))
    plan = P.SourceProductionPlan(basis, source, P.ProductionPurpose.VALUE, rid('root'),
        tuple(regions), tuple(steps), ())
    digest = _digest(encode(plan, record_tags=P.PRODUCTION_RECORD_TAGS))
    origin = SourceTemplateOrigin(SourceMaterialDomain.EXPRESSIONS, 1, digest)
    nodes = {key: SourceTemplateNode(key, rid('root'), (reference, world), (completion(world),), origin),
        child_key: SourceTemplateNode(child_key, rid('child-root'), (reference, world),
            (completion(world),), replace(origin, plan_sha256='0'*64))}
    link = SourceTemplateLink(key, sid('child'), child_key,
        (parameter('root', 0), parameter('root', 1)))
    interfaces = MappingProxyType(dict(nodes=MappingProxyType(nodes),
        links=MappingProxyType({(key, sid('child')): link})))

    emitted = []
    original_run = normal_graph._Emitter.run

    def watched_run(emitter):
        emitted.append(tuple(region.id for region in emitter.plan.regions))
        return original_run(emitter)

    monkeypatch.setattr(normal_graph._Emitter, 'run', watched_run)
    index = RegionProjectionIndex(plan, interfaces)
    assert not emitted and not index.projected_regions
    assert index.interfaces is interfaces
    assert index.owner_region(sid('declare')) == rid('root')
    assert index.owner_region(rid('body')) == rid('body')
    assert index.touching_regions(sid('declare')) == (rid('root'), rid('body'))
    assert index.touching_regions(rid('root')) == (rid('root'), rid('body'))
    assert index.children[rid('root')] == (rid('loop'), rid('deferred'))
    assert index.children[rid('body')] == (rid('yes'), rid('no'))
    assert rid('loop') not in index.touching_regions(rid('root'))
    assert not emitted and not index.projected_regions
    root_packet = index.project(rid('root'))
    assert emitted == [(rid('root'),)]
    assert index.project(rid('root')) is root_packet
    assert root_packet.region is regions[0]
    assert all(step is next(original for original in steps if original.id == step.id)
        for step in root_packet.steps)
    assert any(edge.kind is G.EdgeKind.DECLARE for edge in root_packet.edges)
    assert any(port.template == child_key for port in root_packet.external_ports)
    assert not any(edge.transition is G.Transition.BACKEDGE for edge in root_packet.edges)
    loop_packet = index.project(rid('loop'))
    assert any(edge.transition is G.Transition.BACKEDGE and edge.kind is G.EdgeKind.STATE
        for edge in loop_packet.edges)
    body_packet = index.project(rid('body'))
    assert {edge.when for edge in body_packet.edges if edge.condition is not None} == {True, False}
    packets = index.project_all()
    assert all(len(row) == 1 for row in emitted)
    monkeypatch.setattr(normal_graph._Emitter, 'run', original_run)
    original = tuple(normal_graph.normal_edges(plan, interfaces))
    assert tuple(edge for packet in packets for edge in packet.edges) == original
    for packet in packets:
        endpoints = {port for edge in packet.edges
            for port in (edge.source, edge.target, edge.condition) if port is not None}
        assert set(packet.local_ports).isdisjoint(packet.external_ports)
        assert set(packet.local_ports).union(packet.external_ports) == endpoints
        for port in endpoints:
            if port.template == key:
                assert packet.region.id in index.touching_regions(port.owner)
    with pytest.raises(ValueError, match='sealed template'):
        RegionProjectionIndex(replace(plan, input=replace(basis, f2_sha256='b'*64)), interfaces)

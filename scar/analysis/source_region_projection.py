"""Demand emission of one region's original normal-graph edge packet.

The sealed plan and shared template interfaces remain the authority. A packet
contains its original region/direct steps and preserves external endpoints;
following an edge never emits the target body. Ordered concatenation of every
region's packet is exactly the original normal-graph emission.
"""
from copy import copy
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType

from scar.analysis.normal_graph import _Emitter
from scar.analysis.source_productions import _digest
from scar.ir.program import normal_graph as G, source_productions as P
from scar.ir.program.source_template_links import SourceTemplateKey
from scar.ir.record_codec import encode


@dataclass(frozen=True, slots=True)
class RegionProjection:
    """Original signatures plus edges emitted by these direct steps only.

    ``local_ports``/``external_ports`` partition the emitted edge endpoints,
    including condition ports. Unused declared parameters/completions remain
    present in ``region`` and ``steps`` rather than requiring invented edges.
    Edge tuple positions are local emission ordinals; ``region_ordinal`` gives
    their original order within the complete template's region sequence.
    """
    template: SourceTemplateKey
    plan_sha256: str
    region_ordinal: int
    region: P.ProductionRegion
    steps: tuple[P.ProductionStep, ...]
    edges: tuple[G.Edge, ...]
    local_ports: tuple[G.Port, ...]
    external_ports: tuple[G.Port, ...]


class _RegionCursor:
    """Limit the existing emitter's traversal, without creating another plan."""
    __slots__ = ('original', 'regions')

    def __init__(self, original, region):
        self.original = original
        self.regions = (region,)

    def __getattr__(self, name):
        return getattr(self.original, name)


class RegionProjectionIndex:
    """Validate once, index once, emit only regions explicitly requested.

    The shared emitter's region/step and iterator-exhaustion indexes are made
    over the original plan. Each request gets its own output/cursor, reusing
    ``_Emitter.run`` and its existing call/loop/return/declaration rules. No
    entire-plan edge list or recursively expanded callee body is constructed.
    """

    def __init__(self, plan, interfaces):
        if type(plan) is not P.SourceProductionPlan:
            raise TypeError('region projection needs the original typed source plan')
        key = SourceTemplateKey(plan.source.occurrence, plan.purpose)
        origin = interfaces['nodes'][key].origin
        digest = _digest(encode(plan, record_tags=P.PRODUCTION_RECORD_TAGS))
        if origin.plan_sha256 != digest:
            raise ValueError('region projection input differs from the sealed template')
        self.plan = plan
        self.template = key
        self.plan_sha256 = digest
        self.origin = origin
        # A provider can share its already frozen unit-wide maps across every
        # template index. Do not copy those maps again for each small plan.
        if isinstance(interfaces, MappingProxyType):
            self.interfaces = interfaces
        else:
            self.interfaces = MappingProxyType({name: values
                if isinstance(values, MappingProxyType) else MappingProxyType(dict(values))
                for name, values in ((name, interfaces[name]) for name in ('nodes', 'links'))})
        self._emitter = _Emitter(plan, key, self.interfaces)
        self._ordinals = {region.id: index for index, region in enumerate(plan.regions)}
        self._region_ids = tuple(self._ordinals)
        self._projections = {}
        self._build_routing()

    def _build_routing(self):
        """Index original typed mentions, without running any edge emitter.

        A region can emit an edge touching a foreign region's parameters,
        results or call boundary. References nested inside invocation inputs,
        saved values and captures count equally. Lexical parenthood is kept
        separately; it is not evidence of an invocation or of a value use.
        """
        mentioned_cache, field_cache = {}, {}
        empty = frozenset()

        def mentioned(value):
            kind = type(value)
            if kind in (P.ProductionRegionID, P.ProductionStepID):
                return (value,)
            if kind is P.ProductionParameter:
                return (value.region,)
            if kind is P.ProductionResult:
                return (value.step,)
            if kind is not tuple and not is_dataclass(value):
                return empty
            identity = id(value)
            if identity not in mentioned_cache:
                if kind is tuple:
                    children = value
                else:
                    if kind not in field_cache:
                        field_cache[kind] = tuple(field.name for field in fields(value))
                    children = (getattr(value, name) for name in field_cache[kind])
                mentioned_cache[identity] = frozenset(owner for child in children
                    for owner in mentioned(child))
            return mentioned_cache[identity]

        owners, touching = {}, {}
        children = {region: [] for region in self._region_ids}
        for region in self.plan.regions:
            owners[region.id] = region.id
            if region.parent is not None:
                children[region.parent].append(region.id)
            references = {region.id, *region.steps}
            for identity in region.steps:
                step = self._emitter.steps[identity]
                owners[identity] = region.id
                references.update(mentioned(step.arguments))
                references.update(mentioned(step.instruction))
            for owner in references:
                touching.setdefault(owner, []).append(region.id)
        self._owner_regions = MappingProxyType(owners)
        # Regions were visited once in original plan order, so these lists
        # already are deterministic ordered sets without a sort or emission.
        self._touching_regions = MappingProxyType({owner: tuple(regions)
            for owner, regions in touching.items()})
        self.children = MappingProxyType({owner: tuple(regions)
            for owner, regions in children.items()})

    @property
    def region_ids(self):
        return self._region_ids

    @property
    def projected_regions(self):
        return tuple(self._projections)

    def owner_region(self, owner):
        """Return the original region owning this region/step identity."""
        if type(owner) not in (P.ProductionRegionID, P.ProductionStepID):
            raise TypeError('original typed region or step identity required')
        return self._owner_regions[owner]

    def touching_regions(self, owner):
        """Conservative emitting-region candidates in original plan order."""
        if type(owner) not in (P.ProductionRegionID, P.ProductionStepID):
            raise TypeError('original typed region or step identity required')
        return self._touching_regions.get(owner, ())

    def project(self, region):
        if type(region) is not P.ProductionRegionID:
            raise TypeError('original typed region identity required')
        if region in self._projections:
            return self._projections[region]
        original = self._emitter.regions[region]
        emitter = copy(self._emitter)
        emitter.plan = _RegionCursor(self.plan, original)
        emitter.out = []
        emitter.step = None
        edges = tuple(emitter.run())
        steps = tuple(self._emitter.steps[identity] for identity in original.steps)
        owners = {original.id, *original.steps}
        endpoints = tuple(dict.fromkeys(port for edge in edges
            for port in (edge.source, edge.target, edge.condition) if port is not None))
        local = tuple(port for port in endpoints
            if port.template == self.template and port.owner in owners)
        external = tuple(port for port in endpoints
            if port.template != self.template or port.owner not in owners)
        result = RegionProjection(self.template, self.plan_sha256, self._ordinals[region],
            original, steps, edges, local, external)
        self._projections[region] = result
        return result

    def project_all(self):
        return tuple(self.project(region.id) for region in self.plan.regions)


__all__ = ['RegionProjection', 'RegionProjectionIndex']

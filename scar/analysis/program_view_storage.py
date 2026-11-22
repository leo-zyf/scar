"""Compact unit-local storage of an already admitted source program view.

The owning header supplies the one unit, GraphNamespace and material receipt.
Packing preserves projection order and emits records lazily; it never expands
templates, reconstructs nested namespaces or substitutes source evidence.
"""
from itertools import zip_longest

from scar.artifacts import read_record_table as _table
from scar.ir.program import view as V, source_productions as P, normal_graph as G
from scar.ir.program.dependencies import DependencyPoint, DependencyCondition
from scar.ir.record_codec import _decode, _record_tags


def _local(provider, point):
    if (type(point) is not DependencyPoint or point.context is not None
            or point.namespace != provider.namespace
            or point.namespace.graph != provider.namespace.graph
            or point.namespace.source_body != provider.namespace.source_body):
        raise ValueError('stored source endpoint has a context or different namespace declaration')
    if point.node not in provider.nodes:
        raise ValueError('stored source endpoint is absent from the admitted unit')
    return point.node


def pack_nodes(provider):
    """Yield StoredSourceNode records; exhaust to complete the guard checks."""
    provider.check()
    for identity, row in provider.nodes.items():
        if type(row) is not V.SourceNodeProjection or identity != _local(provider, row.point):
            raise ValueError('stored node differs from its original projection key')
        if row.anchor.unit != provider.unit or row.material != provider.material.result_artifact:
            raise ValueError('stored node has a different unit or material header')
        owner = None if row.owner is None else _local(provider, row.owner)
        yield V.StoredSourceNode(identity, row.anchor.template, row.anchor.original,
            row.kind, row.role, owner, row.value_type)
    provider.check()


def _condition(provider, condition):
    if type(condition) is not DependencyCondition:
        raise TypeError('stored source condition must be an original typed predicate')
    return V.StoredDependencyCondition(condition.domain, condition.name,
        None if condition.subject is None else _local(provider, condition.subject), condition.index,
        tuple(_local(provider, point) for point in condition.operands))


def pack_edges(provider):
    """Yield StoredSourceEdge records with original F4 key/ordinal references."""
    provider.check()
    for relation, projections in provider.edge_sources.items():
        source, target = _local(provider, relation.source), _local(provider, relation.target)
        conditions = tuple(_condition(provider, item) for item in relation.conditions)
        for row in projections:
            if type(row) is not V.SourceEdgeProjection or row.relation != relation:
                raise ValueError('stored edge differs from its owning projected relation')
            if row.original is not None and type(row.original) is not G.Edge:
                raise TypeError('stored original edge must reference a sealed G.Edge')
            if row.original is not None:
                if type(row.region)is not P.ProductionRegionID:
                    raise TypeError('stored original edge needs its original region coordinate')
                packet=provider.region_indexes[row.template].project(row.region)
                if (type(row.ordinal)is not int or not 0<=row.ordinal<len(packet.edges)
                        or packet.edges[row.ordinal]!=row.original):
                    raise ValueError('stored original edge differs from sealed region/local ordinal')
            elif row.region is not None:
                raise ValueError('helper relation cannot impersonate an original region edge')
            # This exact provenance is factored into the shared header and
            # rule/endpoints. Reject any extra source/anchor instead of silently
            # losing evidence that the compact schema cannot represent.
            if (len(relation.evidence) != 1 or relation.evidence[0].rule != row.rule
                    or relation.evidence[0].sources != (provider.locator,)
                    or relation.evidence[0].anchors != (relation.source, relation.target)):
                raise ValueError('stored edge has evidence beyond its shared header/rule/endpoints')
            for point in relation.evidence[0].anchors:
                _local(provider, point)
            yield V.StoredSourceEdge(source, target, relation.kind, relation.meaning, relation.strength,
                row.template, row.ordinal, row.rule, row.original is not None, conditions, row.region)
    provider.check()


def read_and_validate(folder, tables, provider):
    """Strictly compare stored local records to a caller-certified projection.

    The caller validates the shard header and derives provider through its
    ordinary source reader. Both complete table receipts and typed contents
    are checked here without rebuilding DependencyPoints. Return row counts.
    """
    if set(tables) != {'nodes', 'edges'}:
        raise ValueError('compact source view needs exactly nodes and edges tables')
    provider.check(); missing = object(); counts = {}
    # One private validated snapshot per read, just as normal_output prepares
    # tags per write. Every row still uses the original strict typed decoder.
    tags = _record_tags(P.PRODUCTION_RECORD_TAGS)
    for name, cls, expected in (('nodes', V.StoredSourceNode, pack_nodes(provider)),
            ('edges', V.StoredSourceEdge, pack_edges(provider))):
        count = 0
        for wire, record in zip_longest(_table(folder, tables[name]), expected, fillvalue=missing):
            if wire is missing or record is missing:
                raise ValueError('compact source '+name+' table length differs')
            actual = _decode(cls, wire, tags)
            if actual != record:
                raise ValueError('compact source '+name+' record differs from its original projection')
            count += 1
        counts[name] = count
    provider.check()
    return counts

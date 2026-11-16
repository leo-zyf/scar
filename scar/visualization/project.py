"""Bounded, offline presentation of the existing qualified ProgramView graph."""
from collections import deque
from dataclasses import fields, is_dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import time

from scar.ir.record_codec import encode
from scar.ir.program.view import ProgramEntityKind
from scar.ir.program.dependencies import GraphNamespace
from scar.ir.v2.ids import Identifier

SCHEMA = 'scar.graph-viewer-projection'
_LAYER = {**dict.fromkeys(('VALUE_USE','OPERAND','RESULT','PROJECTION','BLOCK_ARGUMENT','BINDING_PAYLOAD','BINDING_READ','ALIAS'), 'data'),
    **dict.fromkeys(('CONTROL','CONTROL_FLOW','CONTROL_CONDITION','SEQUENCE','ORDER'), 'control'),
    **dict.fromkeys(('STATE_INPUT','STATE_OUTPUT','STATE_INTERFACE','STATE_LOCATION','STATE_VALUE'), 'state'),
    **dict.fromkeys(('RESOURCE',), 'resource'),
    **dict.fromkeys(('CONTAINS','IMPLEMENTATION','ENVIRONMENT','FIELD_INITIALIZER'), 'structure'),
    **dict.fromkeys(('CALL_TARGET','ARGUMENT','COMPLETION'), 'call')}


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _key(value):
    # Namespace declarations may carry additional evidence without changing
    # the original graph/endpoint identity; match GraphNamespace.__eq__.
    if type(value) is GraphNamespace:
        return _key(value.identity)
    if isinstance(value, (Identifier, Enum)):
        return encode(value)
    if is_dataclass(value):
        return {f.name: _key(getattr(value, f.name)) for f in fields(value) if f.compare}
    if isinstance(value, (tuple, list)):
        return [_key(v) for v in value]
    if isinstance(value, dict):
        return {k: _key(v) for k, v in value.items()}
    return encode(value)


def _identity(value):
    return sha256(_json(_key(value)).encode()).hexdigest()


def _point_id(point):
    # The complete qualified point, including the original context, is the key.
    return 'n-' + _identity(point)


def _source(original, point):
    result = dict(locator=encode(point.namespace.graph), record_type=type(original).__name__)
    for name in ('id', 'source', 'template', 'owner', 'ordinal', 'completion', 'path'):
        value = getattr(original, name, None)
        if value is not None:
            result[name] = encode(value)
    return result


def _label(original, node):
    instruction = getattr(original, 'instruction', None)
    if instruction is not None:
        name = type(instruction).__name__
        labels = []
        for field in ('kind', 'action', 'name', 'operator', 'value'):
            value = getattr(instruction, field, None)
            if isinstance(value, Enum):
                value = value.value
            if type(value) in (str, int, float, bool):
                labels.append(field + '=' + str(value)[:80])
        return name + (':' + ', '.join(labels) if labels else '')
    kind = getattr(original, 'kind', None)
    return str(getattr(kind, 'value', kind) or node.role.value) + ' ' + str(node.point.node)[-10:]


def _details(original, node):
    record = encode(original); size = len(_json(record).encode())
    return dict(original=_source(original, node.point), record=record if size <= 16384 else None,
        raw_record_bytes=size, raw_record_truncated=size > 16384, value_type=encode(node.value_type))


def project_view(view, roots, *, max_nodes_per_part=400, max_parts=32, max_depth=3,
                 max_edges_per_part=1600, max_seconds=60):
    """Read bounded original scopes, without unrolling loops or claiming closure.

    Every part is a presentation packet. A child link is an original expand or
    CONTAINS relation, never a fabricated dependency. Unloaded scopes and limits
    remain explicit boundaries. Time limits are cooperative query boundaries.
    """
    for name, value in (('max_nodes_per_part', max_nodes_per_part), ('max_parts', max_parts),
                        ('max_edges_per_part', max_edges_per_part)):
        if type(value) is not int or value < 1:
            raise ValueError(name + ' must be a positive integer')
    if type(max_depth) is not int or max_depth < 0 or type(max_seconds) not in (int, float) or max_seconds <= 0:
        raise ValueError('nonnegative depth and positive cooperative time budget required')
    roots = tuple(dict.fromkeys(roots))
    if not roots:
        raise ValueError('at least one original qualified root required')
    started = time.monotonic(); pending = deque(); scheduled = {}; parts = []; global_boundaries = []

    def schedule(point, depth):
        identity = 'p-' + _identity(point)
        if identity in scheduled:
            return identity
        if depth > max_depth or len(scheduled) >= max_parts:
            return None
        scheduled[identity] = point; pending.append((point, depth, identity))
        return identity

    root_parts = []
    for point in roots:
        identity = schedule(point, 0)
        if identity is None:
            global_boundaries.append(dict(kind='MAX_PARTS', point=encode(point)))
        else:
            root_parts.append(identity)
    while pending:
        if time.monotonic() - started >= max_seconds:
            global_boundaries.extend(dict(kind='TIME_BUDGET_UNEXPANDED_SCOPE', part=identity,
                point=encode(point)) for point, _, identity in pending)
            break
        scope, depth, identity = pending.popleft()
        nodes = {}; edges = {}; frontiers = {}; boundaries = []; boundary_counts = {}; queries = deque(); queried = set()

        def boundary(kind, point=None, **details):
            boundary_counts[kind] = boundary_counts.get(kind, 0) + 1
            if len(boundaries) >= max_nodes_per_part:
                return
            row = dict(kind=kind, **details)
            if point is not None:
                row['point'] = encode(point)
            boundaries.append(row)

        def add(point, *, query=False):
            key = _point_id(point)
            if key not in nodes:
                if len(nodes) >= max_nodes_per_part:
                    return None
                node = view.get(point); original = view.original(point)
                nodes[key] = dict(id=key, label=_label(original, node),
                    kind=None if node.kind is None else node.kind.value, role=node.role.value,
                    owner=None if node.owner is None else _point_id(node.owner), point=encode(point),
                    source=_source(original, point), child_parts=[], collapsed=True,
                    original_kind=getattr(getattr(original, 'kind', None), 'value', None),
                    details=_details(original, node))
            if query and key not in queried:
                queries.append(point)
            return key

        def child(point):
            key = add(point)
            if key is None:
                boundary('MAX_NODES', point)
                return
            target = schedule(point, depth + 1)
            if target is None:
                boundary('UNEXPANDED_SCOPE', point, reason='MAX_DEPTH' if depth >= max_depth else 'MAX_PARTS')
            elif target != identity and target not in nodes[key]['child_parts']:
                nodes[key]['child_parts'].append(target)

        add(scope, query=True)
        children = view.expand(scope)
        for point in children:
            key = add(point, query=True)
            if key is None:
                boundary('MAX_NODES', point)
                continue
            node = view.get(point)
            if node.kind is ProgramEntityKind.REGION:
                child(point)
            elif node.kind is ProgramEntityKind.OPERATION:
                for port in view.expand(point):
                    if view.get(port).kind is ProgramEntityKind.REGION:
                        child(port)
                    elif add(port, query=True) is None:
                        boundary('MAX_NODES', port)
        while queries:
            if time.monotonic() - started >= max_seconds:
                boundary('TIME_BUDGET', remaining_queries=len(queries))
                break
            point = queries.popleft(); key = _point_id(point)
            if key in queried:
                continue
            queried.add(key)
            for direction, query in (('incoming', view.producers), ('outgoing', view.consumers)):
                neighborhood = query(point)
                if not neighborhood.enumerated:
                    boundary('NEIGHBORHOOD_NOT_ENUMERATED', point, direction=direction)
                for row in neighborhood.frontiers:
                    encoded = encode(row); frontiers[_json(encoded)] = encoded
                for edge in neighborhood.edges:
                    edge_id = 'e-' + _identity(edge)
                    if edge_id in edges:
                        continue
                    if len(edges) >= max_edges_per_part:
                        boundary('MAX_EDGES', point, direction=direction)
                        break
                    missing = [p for p in (edge.source, edge.target) if _point_id(p) not in nodes]
                    if len(nodes) + len(set(missing)) > max_nodes_per_part:
                        boundary('MAX_NODES_EDGE_ENDPOINTS', point, edge=edge_id,
                                 endpoints=[encode(p) for p in missing])
                        continue
                    for endpoint in missing:
                        add(endpoint)
                    edges[edge_id] = dict(id=edge_id, source=_point_id(edge.source), target=_point_id(edge.target),
                        kind=edge.kind.value, layer=_LAYER.get(edge.kind.value, 'other'),
                        meaning=edge.meaning.value, strength=edge.strength.value,
                        conditions=encode(edge.conditions), evidence=encode(edge.evidence),
                        runtime_occurrence=False)
                    if edge.kind.value == 'CONTAINS' and edge.source == scope:
                        if view.get(edge.target).kind is ProgramEntityKind.REGION:
                            child(edge.target)
        # Endpoint-only nodes are visible but have not had their neighborhood queried.
        for key, node in nodes.items():
            node['neighborhood_queried'] = key in queried
            if key not in queried:
                boundary('ENDPOINT_NEIGHBORHOOD_NOT_QUERIED', point=None, node=key)
        parts.append(dict(id=identity, root=_point_id(scope), scope=encode(scope),
            title=nodes[_point_id(scope)]['label'], depth=depth,
            nodes=list(nodes.values()), edges=list(edges.values()), frontiers=list(frontiers.values()),
            boundaries=boundaries, boundary_counts=boundary_counts, complete=False, loops_unrolled=False))
    available = {part['id'] for part in parts}
    for part in parts:
        for node in part['nodes']:
            missing = [p for p in node['child_parts'] if p not in available]
            if missing:
                part['boundaries'].append(dict(kind='CHILD_PART_NOT_EXPORTED', node=node['id'], parts=missing))
                counts = part['boundary_counts']; counts['CHILD_PART_NOT_EXPORTED'] = counts.get('CHILD_PART_NOT_EXPORTED', 0) + 1
                node['child_parts'] = [p for p in node['child_parts'] if p in available]
    return dict(schema=SCHEMA, schema_version=1, root=root_parts[0] if root_parts else None,
        title='SCAR bounded program graph', roots=root_parts, parts=parts, boundaries=global_boundaries,
        limits=dict(max_nodes_per_part=max_nodes_per_part, max_parts=max_parts, max_depth=max_depth,
            max_edges_per_part=max_edges_per_part, max_seconds=max_seconds,
            time_enforcement='COOPERATIVE_QUERY_BOUNDARIES'),
        coverage=dict(complete=False, scope='BOUNDED_QUERIED_ORIGINAL_SCOPES',
            parts=len(parts), nodes=sum(len(p['nodes']) for p in parts), edges=sum(len(p['edges']) for p in parts),
            loops_unrolled=False, execution_proof=False, target_program_executed=False),
        stats=dict(parts=len(parts), nodes=sum(len(p['nodes']) for p in parts),
            edges=sum(len(p['edges']) for p in parts)), seconds=time.monotonic() - started)


def project_build(build_path, *, max_nodes_per_part=400, max_parts=32, max_depth=3,
                 max_edges_per_part=1600, max_seconds=60):
    """Open a strictly admitted ProgramBuild and return bounded presentation packets."""
    from scar.analysis.build import BuildSession
    from scar.ir.execution_recipe import artifact_reference, read_artifact
    build_path = Path(build_path).resolve()
    if build_path.is_file():
        build_path = build_path.parent
    reference = artifact_reference(build_path / 'manifest.json', 'scar.program-build')
    with BuildSession(build_path, seconds=max_seconds) as session:
        result = project_view(session.view, (session.entry,), max_nodes_per_part=max_nodes_per_part,
            max_parts=max_parts, max_depth=max_depth, max_edges_per_part=max_edges_per_part,
            max_seconds=max_seconds)
        read_artifact(reference)
        result['source'] = dict(kind='PROGRAM_BUILD', reference=reference, entry=encode(session.entry))
    result['title'] = Path(session.report['entry']).name + ' — SCAR program graph'
    return result


def export_build(build_path, out, **bounds):
    """Optional JSON-only publisher; the UI bundle uses project_build directly."""
    directory = Path(out).resolve()
    if directory.exists():
        raise ValueError('fresh viewer projection directory required')
    result = project_build(build_path, **bounds)
    directory.mkdir(parents=True)
    rows = []
    for part in result['parts']:
        name = part['id'] + '.json'
        path = directory / name
        path.write_text(_json(part) + '\n')
        rows.append(dict(id=part['id'], path=name, root=part['root'], title=part['title'],
            node_count=len(part['nodes']), edge_count=len(part['edges']),
            boundary_count=len(part['boundaries']), frontier_count=len(part['frontiers']), depth=part['depth']))
    result['parts'] = rows
    (directory / 'manifest.json').write_text(_json(result) + '\n')
    return result


__all__ = ['project_view', 'project_build', 'export_build']

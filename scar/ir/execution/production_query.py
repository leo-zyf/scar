"""Production query."""

from __future__ import annotations

from copy import deepcopy
from scar.ir.provenance import _byte_bounds, _width
from scar.ir.v2.evidence import EvidenceNodeKind, EvidenceRelation
from scar.ir.v2.values import ValueGraph
from scar.ir.execution.common import Unresolved
from scar.ir.execution.content import (
    EndpointContent,
    ObservationConsumer,
    VersionRef,
    NativeOutputRef,
    ProtocolSelector,
    StorageContent,
)
from scar.ir.execution.content_validation import artifact_key, point_scope, target_key, native_port, resolve_placement
from scar.ir.execution.coordinates import OccurrencePhase, OccurrencePoint
from scar.ir.execution.coverage import RegionRef, CoverageStatus, _ValueCatalog, _storage_coverage
from scar.ir.execution.common import ContentReadID, OperationOccurrenceID
from scar.ir.execution.production import (
    InventorySourceRef,
    RawRecordRef,
    SourceEdgeRef,
    SourceEndpointRef,
    BytePrefixWrite,
    ExternalEstablishment,
    HostWrite,
    StateProduction,
    ValueProduction,
    WorkWrite,
)
from scar.ir.execution.endpoint_protocol import (
    same_materialization,
    endpoint_extent,
    resolve_endpoint_profile,
    source_byte_input,
)
from scar.ir.execution.source_validation import resolve_endpoint
from scar.ir.execution.work import MaterializationRef, WithinOperation
from scar.ir.execution.validity_budget import ValidityBudget
from scar.ir.program.model import (
    AccessMode,
    DataValue,
    ReferenceValue,
    ExactLiteralOrigin,
    LiteralOrigin,
)
from scar.ir.v2.common import Completeness
from scar.ir.execution.effects import effect_context, resolve_canonical_effect
from scar.ir.literals import LiteralKind
from scar.ir.program.providers import ProviderFactReference


# Source-rooted production inspection, without a closed writer set or ready proof.
#
# The traversal budget counts lineage nodes, source edges/raw rows, inspected
# association joins, and selector ranks.  Snapshot validation and inventory replay
# have their own byte/line budgets and happen before this query; they are not
# represented as cheap traversal operations.  No URI or provider is executed here.
_RELATIONS = frozenset((EvidenceRelation.PRODUCES, EvidenceRelation.MATERIALIZES,
    EvidenceRelation.OBSERVED_WRITE, EvidenceRelation.OVERWRITES,
    EvidenceRelation.ALIASES, EvidenceRelation.USES_STORAGE, EvidenceRelation.ESCAPES))


class _Budget:
    def __init__(self, limit, gaps):
        if type(limit) is not int:
            raise TypeError('production budget must be an exact integer')
        if limit <= 0:
            raise ValueError('production budget must be positive')
        self.limit, self.used, self.truncated = limit, 0, False
        self.costs = {key: 0 for key in ('lineage_nodes', 'source_edges', 'raw_rows',
                                        'association_joins', 'selector_rank')}
        self.gaps = gaps

    def take(self, kind, path, count=1):
        if count > self.limit - self.used:
            if not self.truncated:
                self.gaps.append({'code': 'production_budget', 'path': path,
                    'message': 'traversal truncated; unvisited sources/joins are not independent or absent'})
            self.truncated = True
            return False
        self.used += count
        self.costs[kind] += count
        return True


class _SharedBudget:
    """P traversal charged immediately to the caller's one derived budget."""
    def __init__(self, shared, gaps):
        self.shared, self.gaps = shared, gaps
        self.used, self.truncated = 0, False
        self.costs = {key: 0 for key in ('lineage_nodes', 'source_edges', 'raw_rows',
                                        'association_joins', 'selector_rank')}

    def take(self, kind, path, count=1):
        before = self.shared.used
        accepted = self.shared.spend('production_query', count, path=path)
        charged = self.shared.used - before
        self.used += charged
        self.costs[kind] += charged
        if not accepted:
            if not self.truncated:
                _gap(self.gaps, 'production_budget', path,
                     'traversal truncated; unvisited sources/joins are not independent or absent')
            self.truncated = True
        return accepted


def _gap(gaps, code, path, message):
    gaps.append({'code': code, 'path': path, 'message': message})


def _version_key(reference):
    return artifact_key(reference.artifact), reference.version


def _lineage(ctx, demand, budget, gaps):
    if demand is None:
        _gap(gaps, 'production_lineage_unresolved', 'demand', 'no resolved scoped demand version')
        return (), set()
    pending, seen, result = [demand], set(), []
    while pending:
        ref = pending.pop()
        key = _version_key(ref)
        if key in seen:
            continue
        if not budget.take('lineage_nodes', 'lineage'):
            break
        seen.add(key)
        graph = ctx.snapshots.graph(ref.artifact, ValueGraph)
        row = None if graph is None else graph.versions.get(ref.version)
        if row is None:
            _gap(gaps, 'production_lineage_pending', 'lineage', 'original version graph/row is unavailable')
            continue
        provenance = None if row.provenance_id is None else graph.provenance.get(row.provenance_id)
        parents, seen_parents = [], set()
        dependencies_complete = True
        for references in (row.parent_versions, () if provenance is None else provenance.inputs):
            for parent in references:
                if not budget.take('association_joins', 'lineage.dependencies'):
                    dependencies_complete = False
                    break
                if parent not in seen_parents:
                    parents.append(parent)
                    seen_parents.add(parent)
            if not dependencies_complete:
                break
        dependencies = tuple(VersionRef(ref.artifact, item) for item in parents)
        result.append({'version': ref, 'original': row, 'provenance': provenance,
                       'dependencies': dependencies, 'dependencies_complete': dependencies_complete,
                       'is_demand': key == _version_key(demand)})
        pending.extend(reversed(dependencies))
    _gap(gaps, 'production_lineage_unverified', 'lineage',
         'declared parent/provenance dependencies are not native production, purity, or current writers')
    return tuple(result), seen


def _run(ctx, resolved):
    read = resolved['read']
    if type(read.consumer) is not ObservationConsumer:
        return point_scope(ctx, OccurrencePoint(read.consumer.subject.operation, OccurrencePhase.BEFORE))[0]
    root = resolved['observation_root']
    if root is not None:
        for point in (root.interval.start, root.interval.end, root.anchor):
            run = point_scope(ctx, point)[0]
            if run is not None:
                return run
    return None


def _universe(ctx, run, budget, gaps):
    """Enumerate first: association deletion cannot buy a smaller source universe."""
    candidates, pending = [], []
    inventories = tuple(row for row in ctx.source_inventories.values() if run is None or row.run == run)
    if not inventories:
        _gap(gaps, 'production_inventory_missing', 'inventory',
             'no independent source inventory for the original consumer run; an empty AND is not satisfied')
    if run is None:
        _gap(gaps, 'production_run_unresolved', 'demand',
             'consumer run is unresolved; all inventories remain candidates, without a scope match proof')
    for inventory in inventories:
        resolution = ctx.source_resolutions.get(inventory.id)
        if resolution is None or not resolution.sources:
            pending.append({'inventory': inventory, 'source': None, 'reason': 'source inventory/index pending'})
            continue
        for declared in resolution.sources:
            source = ctx.inventory_sources.get((inventory.id, declared.key), declared)
            reference = InventorySourceRef(inventory.id, source.key)
            if source.bundle is None or not source.projection_replayed:
                pending.append({'inventory': inventory, 'source': reference,
                                'reason': 'source projection is pending; raw candidates do not close its scope'})
            if source.bundle is not None:
                for edge in source.bundle.evidence.edges.values():
                    if not budget.take('source_edges', f'source[{reference}].edges'):
                        pending.append({'inventory': inventory, 'source': reference,
                                        'reason': 'unvisited source edges after traversal budget'})
                        break
                    if edge.relation in _RELATIONS:
                        candidates.append({'id': ('edge', inventory.id, source.key, edge.edge_id),
                            'source': SourceEdgeRef(reference, edge.edge_id), 'original': edge,
                            'relation': edge.relation.value, 'endpoints': (),
                            'write_associations': (), 'outcome_productions': (),
                            'relevance': (), 'conditions': []})
            for raw in source.raw_records:
                if not budget.take('raw_rows', f'source[{reference}].raw'):
                    pending.append({'inventory': inventory, 'source': reference,
                                    'reason': 'unvisited raw rows after traversal budget'})
                    break
                metadata = raw.payload.get('metadata') if type(raw.payload) is dict else None
                native = type(metadata) is dict and 'scar_native_result' in metadata
                # Even a non-native row can contain an unprojected effect. Keep
                # it as raw evidence; do not invent an ALIASES/OVERWRITES edge.
                candidates.append({'id': ('raw', inventory.id, source.key, raw.line, raw.sha256),
                    'source': RawRecordRef(reference, raw.line, raw.sha256), 'original': raw,
                    'relation': 'raw_native_result' if native else 'raw_record', 'endpoints': (),
                    'write_associations': (), 'outcome_productions': (),
                    'relevance': (), 'conditions': []})
    if pending:
        _gap(gaps, 'production_inventory_pending', 'inventory',
             'unresolved or unvisited source scope remains; enumerated candidates are not a closed writer inventory')
    return candidates, tuple(pending)


def _facets(ctx, resolved):
    """Derive original facets; bare identifiers never cross artifact scopes."""
    artifact = resolved.get('values_artifact')
    if artifact is None:
        return None
    return {'artifact': artifact,
        **{key: resolved.get(key) for key in ('value_version', 'materialization', 'region', 'allocation')}}


def _target_facets(resolved):
    graph, artifact = resolved['graph'], resolved['artifact']
    result = {'artifact': artifact, 'value_version': None, 'materialization': None,
              'region': None, 'allocation': None}
    result[resolved['facet']] = resolved['identity']
    if result['materialization'] is not None:
        mat = graph.materializations[result['materialization']]
        result.update(value_version=mat.value_version, region=mat.region)
    if result['region'] is not None:
        result['allocation'] = graph.regions[result['region']].allocation
    return result


def _endpoints(ctx, candidate, budget, gaps):
    ref, original = candidate['source'], candidate['original']
    if type(ref) is not SourceEdgeRef:
        return ()
    result = []
    for endpoint in (original.source, original.target):
        if endpoint.kind is EvidenceNodeKind.OPERATION_INSTANCE:
            continue
        if not budget.take('association_joins', f'{candidate["id"]}.endpoint'):
            break
        reference = SourceEndpointRef(ref.source, endpoint)
        resolved = resolve_endpoint(ctx, reference, f'production[{candidate["id"]}].endpoint')
        mappings = []
        for bridge in ctx.resolved_source_bridges.values():
            if not budget.take('association_joins', f'{candidate["id"]}.bridge'):
                break
            if bridge['row'].endpoint == reference:
                mappings.append(bridge)
        if len(mappings) > 1:
            candidate['conditions'].append('multiple_endpoint_bridges')
            _gap(gaps, 'production_bridge_ambiguous', str(candidate['id']),
                 'multiple declared bridge targets are retained; no first target is chosen')
        result.append({'reference': reference, 'resolved': resolved, 'bridges': tuple(mappings)})
    return tuple(result)


def _spatial(ctx, facets, demand, budget, path):
    endpoint = demand.get('endpoint_extent')
    if type(endpoint) is dict and endpoint.get('kind') == 'endpoint-byte-range':
        other = facets
        if type(other) is not dict or other.get('kind') != 'endpoint-byte-range':
            return 'unresolved'
        if not budget.take('selector_rank', path, 1):
            return 'unresolved'
        if target_key(endpoint['target']) != target_key(other['target']):
            # Distinct declared ResourceInstances need not be physically
            # disjoint; retain a possible alias until a scoped native rule can
            # establish that fact.
            return 'possible_overlap'
        left_start, left_end = endpoint['offset_bytes'], endpoint['offset_bytes'] + endpoint['length_bytes']
        right_start, right_end = other['offset_bytes'], other['offset_bytes'] + other['length_bytes']
        if left_start == left_end or right_start == right_end:
            return 'disjoint'
        return 'disjoint' if left_end <= right_start or right_end <= left_start else 'overlap'
    region = demand.get('selector')
    if type(region) is not RegionRef:
        return 'unresolved'
    if artifact_key(facets['artifact']) != artifact_key(region.artifact):
        return 'different_scope'
    graph = ctx.snapshots.graph(region.artifact, ValueGraph)
    if graph is None:
        return 'unresolved'
    required = graph.regions[region.region]
    allocation = facets.get('allocation')
    if allocation is None:
        return 'unresolved'
    if allocation != required.allocation:
        return 'different_allocation'
    offered = None if facets.get('region') is None else graph.regions[facets['region']]
    if offered is None:
        return 'possible_overlap'
    if not budget.take('selector_rank', path, len(required.shape) + len(offered.shape)):
        return 'unresolved'
    if 0 in required.shape or 0 in offered.shape:
        return 'disjoint'
    if (_width(required) is None or _width(offered) is None
            or required.layout != 'strided' or offered.layout != 'strided'):
        return 'possible_overlap'
    if offered.id == required.id:
        return 'overlap'
    left, right = _byte_bounds(required), _byte_bounds(offered)
    if left[0] >= right[1] or right[0] >= left[1]:
        return 'disjoint'
    # Bounding boxes cannot fill holes. This is a possible conflict only.
    return 'possible_overlap'


def _relevance(ctx, candidate, demand, lineage, budget):
    labels, spatial = set(), []
    for endpoint in candidate['endpoints']:
        base = _facets(ctx, endpoint['resolved']) if endpoint['resolved'] is not None else None
        variants = ([] if base is None else [(base, False)])
        for bridge in endpoint['bridges']:
            if not budget.take('association_joins', f'{candidate["id"]}.bridge_facets'):
                break
            if bridge['target'] is not None:
                variants.append((_target_facets(bridge['target']), True))
        for facets, conditional in variants:
            if not budget.take('association_joins', f'{candidate["id"]}.facet_relevance'):
                break
            version = facets.get('value_version')
            key = (artifact_key(facets['artifact']), version)
            if version is not None and key in lineage:
                labels.add('lineage_prerequisite')
                if demand['version'] is not None and key == _version_key(demand['version']):
                    labels.add('target_version_observation')
            overlap = _spatial(ctx, facets, demand, budget, str(candidate['id']))
            spatial.append(overlap)
            if overlap in ('overlap', 'possible_overlap'):
                labels.add('possible_invalidator')
            if conditional:
                candidate['conditions'].append('endpoint_bridge_runtime_unverified')
    current = []
    for write in candidate['write_associations']:
        if not budget.take('association_joins', f'{candidate["id"]}.placement'):
            break
        placement, original = write['placement'], write['row'].placement
        demand_placement = demand.get('placement')
        if (type(original) is EndpointContent and type(demand_placement) is EndpointContent
                and type(write.get('endpoint_request')) is dict
                and type(demand.get('endpoint_extent')) is dict):
            request = write['endpoint_request']
            extent = demand['endpoint_extent']
            overlap = _spatial(ctx, request, {'endpoint_extent': extent}, budget,
                               str(candidate['id']))
            if overlap in ('overlap', 'possible_overlap'):
                labels.add('possible_invalidator')
            if placement is None or placement[0] is None:
                candidate['conditions'].append('endpoint_write_materialization_unresolved')
                continue
            mat, _selector = placement
            same_version = (demand['version'] is not None
                and (artifact_key(original.materialization.artifact), mat.value_version)
                    == _version_key(demand['version']))
            same_mat = same_materialization(original.materialization,
                                             demand_placement.materialization)
            same_target = target_key(original.target) == target_key(demand_placement.target)
            if same_version and same_mat and same_target:
                current.append(write)
                candidate['conditions'].append(
                    'endpoint request association is conditional; actual byte coverage and visibility remain separate')
            elif overlap not in ('disjoint', 'different_allocation'):
                candidate['conditions'].append('endpoint_write_demand_relevance_unresolved')
            continue
        if placement is None or type(original) is Unresolved or placement[0] is None:
            candidate['conditions'].append('write_placement_unresolved')
            continue
        mat, selector = placement
        facets = {'artifact': original.materialization.artifact,
            'value_version': mat.value_version, 'materialization': original.materialization.materialization,
            'region': selector.region if type(selector) is RegionRef else mat.region, 'allocation': None}
        graph = ctx.snapshots.graph(facets['artifact'], ValueGraph)
        if graph is not None and facets['region'] is not None:
            facets['allocation'] = graph.regions[facets['region']].allocation
        overlap = _spatial(ctx, facets, demand, budget, str(candidate['id']))
        if overlap in ('overlap', 'possible_overlap'):
            labels.add('possible_invalidator')
        same_version = (demand['version'] is not None
            and (artifact_key(facets['artifact']), mat.value_version) == _version_key(demand['version']))
        same_mat = (demand['materialization'] is not None
            and artifact_key(facets['artifact']) == artifact_key(demand['materialization'].artifact)
            and facets['materialization'] == demand['materialization'].materialization)
        demand_selector = demand['selector']
        same_selector = (type(selector) is RegionRef and type(demand_selector) is RegionRef
            and artifact_key(selector.artifact) == artifact_key(demand_selector.artifact)
            and selector.region == demand_selector.region)
        if type(selector) is not RegionRef and type(demand_selector) is not RegionRef:
            same_selector = original == demand['placement']
        if same_version and same_mat and same_selector:
            current.append(write)
        elif overlap not in ('disjoint', 'different_allocation'):
            candidate['conditions'].append('write_demand_relevance_unresolved')
    candidate['current_write_associations'] = tuple(current)
    if current:
        labels.add('current_production_association')
        candidate['conditions'].append('association_does_not_establish_demand_coverage_or_content')
    if not labels or any(item in ('unresolved', 'different_scope') for item in spatial):
        labels.add('unresolved_relevance')
    # An observed result could be a view/copy/no-op; it is never a native write
    # merely because the source relation is PRODUCES.
    if candidate['relation'] == EvidenceRelation.PRODUCES.value:
        candidate['conditions'].append('observed_result_is_not_native_write')
    if candidate['relation'] == EvidenceRelation.ESCAPES.value:
        labels.add('unresolved_relevance')
        candidate['conditions'].append('escape_effect_mapping_unresolved')
    if type(candidate['source']) is RawRecordRef:
        labels.add('unresolved_relevance')
        candidate['conditions'].append('raw_effect_or_selector_mapping_unresolved')
    candidate['spatial_relations'] = tuple(spatial)
    return tuple(sorted(labels))


def _same_source(left, right):
    if type(left) is RawRecordRef and type(right) is RawRecordRef:
        return (left.source == right.source and left.line == right.line
                and left.sha256.lower() == right.sha256.lower())
    return type(left) is SourceEdgeRef and type(right) is SourceEdgeRef and left == right


def _associations(ctx, candidate, budget, gaps):
    writes, produced = [], []
    for resolved in ctx.resolved_writes.values():
        row = resolved['row']
        matches = False
        for reference in (row.stage, *row.evidence):
            if not budget.take('association_joins', f'{candidate["id"]}.write'):
                break
            if _same_source(reference, candidate['source']):
                matches = True
        if matches:
            writes.append(resolved)
        if budget.truncated:
            break
    write_ids = {item['row'].id for item in writes}
    for resolved in ctx.resolved_produced.values():
        if not budget.take('association_joins', f'{candidate["id"]}.produced'):
            break
        if resolved['row'].write in write_ids:
            produced.append(resolved)
    candidate['write_associations'], candidate['outcome_productions'] = tuple(writes), tuple(produced)
    if not writes:
        candidate['conditions'].append('missing_exact_write_join')
        _gap(gaps, 'production_write_join_missing', str(candidate['id']),
             'no exact SourceEdgeRef or physical raw line/SHA stage connects this source fact to a write')
    elif write_ids - {item['row'].write for item in produced}:
        candidate['conditions'].append('missing_exact_outcome_production_join')
        _gap(gaps, 'production_outcome_join_missing', str(candidate['id']),
             'write association has no exact produced-selector/outcome join; host establishment may have no work outcome')


def _groups(candidates, budget, gaps):
    """Only an identical physical raw stage AND production/placement can group."""
    groups, by_write = [], {}
    for candidate in candidates:
        for write in candidate['current_write_associations']:
            if not budget.take('association_joins', 'current_production.write'):
                break
            row = write['row']
            group = by_write.get(row.id)
            if group is None and type(row.stage) is RawRecordRef:
                for item in groups:
                    if not budget.take('association_joins', 'current_production.group'):
                        break
                    if (type(item['stage']) is RawRecordRef and _same_source(item['stage'], row.stage)
                            and item['production'] == row.production and item['placement'] == row.placement
                            and item['writer'] == row.writer):
                        group = item
                        break
            if group is None:
                group = {'stage': row.stage, 'production': row.production, 'placement': row.placement,
                         'writer': row.writer, 'write_associations': [], 'outcome_productions': [],
                         'candidates': [], '_write_ids': set(), '_produced_ids': set(), '_candidate_ids': set(),
                         'conditions': ('source/native applicability and demand coverage remain unverified',)}
                groups.append(group)
            by_write[row.id] = group
            if row.id not in group['_write_ids']:
                group['write_associations'].append(write)
                group['_write_ids'].add(row.id)
            if candidate['id'] not in group['_candidate_ids']:
                group['candidates'].append(candidate['id'])
                group['_candidate_ids'].add(candidate['id'])
            for item in candidate['outcome_productions']:
                if not budget.take('association_joins', 'current_production.produced'):
                    break
                if item['row'].write == row.id and item['row'].id not in group['_produced_ids']:
                    group['outcome_productions'].append(item)
                    group['_produced_ids'].add(item['row'].id)
    if len(groups) > 1:
        _gap(gaps, 'production_duplicate_stage_unresolved', 'current_production',
             'distinct source stages/production/placement remain separate; shared host, ticket or ancestry does not prove duplicate work')
    return tuple({**{key: value for key, value in group.items() if not key.startswith('_')},
        **{key: tuple(group[key]) for key in
        ('write_associations', 'outcome_productions', 'candidates')}} for group in groups)


def _production(ctx, read_id, traversal, gaps, *, include_context_gaps):
    if type(read_id) is not ContentReadID:
        raise TypeError('production read must be an exact ContentReadID')
    resolved = ctx.resolved_content_reads[read_id]
    read, placement = resolved['read'], resolved['placement']
    materialization = None if type(read.placement) is Unresolved else read.placement.materialization
    version = (None if placement is None or placement[0] is None or materialization is None
               else VersionRef(materialization.artifact, placement[0].value_version))
    endpoint_extent = resolved.get('endpoint_extent')
    demand = {'read': read, 'run': _run(ctx, resolved), 'placement': read.placement,
              'materialization': materialization, 'version': version,
              'selector': endpoint_extent if endpoint_extent is not None else
                          None if placement is None else placement[1],
              'endpoint_extent': endpoint_extent,
              'denotation': resolved['denotation'], 'state_support': read.support}
    lineage, lineage_keys = _lineage(ctx, version, traversal, gaps)
    candidates, pending = _universe(ctx, demand['run'], traversal, gaps)
    # All independent source rows are enumerated before any optional write or
    # bridge joins. Deleting associations cannot shrink this candidate prefix.
    for candidate in candidates:
        candidate['endpoints'] = _endpoints(ctx, candidate, traversal, gaps)
        _associations(ctx, candidate, traversal, gaps)
        candidate['relevance'] = _relevance(ctx, candidate, demand, lineage_keys, traversal)
        if traversal.truncated:
            candidate['conditions'].append('traversal_incomplete')
        candidate['conditions'] = tuple(dict.fromkeys(candidate['conditions']))
    groups = _groups(candidates, traversal, gaps)
    _gap(gaps, 'production_capture_unverified', 'inventory',
         'format replay accounts for supplied files, not all runtime writes, aliases, dropped events or external effects')
    context_gaps = (tuple({'code': item.code, 'path': item.path, 'message': item.message}
                          for item in ctx.gaps) if include_context_gaps else ())
    return {'read': read, 'demand': demand, 'lineage_prerequisites': lineage,
        'candidates': tuple(candidates), 'current_production': groups,
        'possible_invalidators': tuple(item['id'] for item in candidates if 'possible_invalidator' in item['relevance']),
        'unresolved_relevance': tuple(item['id'] for item in candidates if 'unresolved_relevance' in item['relevance']),
        'pending_sources': pending,
        'gaps': context_gaps + tuple(gaps),
        'budget_used': traversal.used, 'budget_costs': traversal.costs, 'truncated': traversal.truncated,
        'budget_scope': 'lineage/source rows/association joins/selector ranks; excludes snapshot/replay byte/line costs and detached report copying',
        'source_runtime_verified': False, 'readiness_verified': False, 'transformation_authorized': False}


def production_shared(ctx, read_id, *, budget: ValidityBudget, include_context_gaps=False):
    """Internal, undetached P view using the whole composition's budget.

    A caller must detach its aggregate once before publishing it. An exhausted
    budget retains pending-source gaps rather than restarting a local allowance.
    """
    if not isinstance(budget, ValidityBudget):
        raise TypeError('production_shared requires a ValidityBudget')
    gaps = []
    traversal = _SharedBudget(budget, gaps)
    if budget.remaining == 0:
        budget.mark_truncated('production', 'no shared budget remains for this read')
        traversal.truncated = True
        _gap(gaps, 'production_budget', 'production', 'read traversal was not started; its obligations remain unresolved')
    return _production(ctx, read_id, traversal, gaps, include_context_gaps=include_context_gaps)


def production(ctx, read_id, *, budget=4096):
    """Return the legacy detached conditional source view for one read."""
    gaps = []
    traversal = _Budget(budget, gaps)
    return deepcopy(_production(ctx, read_id, traversal, gaps, include_context_gaps=True))


# Conditional original write/result associations; no readiness authority.
def _effect(ctx, effect, path):
    """Resolve an original effect; write policy and WorkWrite ownership stay local."""
    issue_codes = {
        'effect_operation': 'production_effect',
        'effect_access': 'production_effect',
        'invoke_kind': 'production_invoke',
        'invoke_input': 'production_invoke_input',
        'invoke_successor': 'production_invoke_successor',
        'invoke_actual_edge': 'production_invoke_actual',
        'invoke_formal': 'production_invoke_formal',
        'invoke_result': 'production_invoke_result',
        'invoke_formal_visit': 'production_invoke_actual',
        'invoke_formal_scope': 'production_invoke_scope',
    }
    gap_codes = {
        'invoke_input_unobserved': 'production_invoke_input_unobserved',
        'invoke_completion_unobserved': 'production_invoke_completion_unobserved',
        'invoke_post_unresolved': 'production_invoke_post_unresolved',
    }
    view = resolve_canonical_effect(effect_context(
        ctx,
        issue=lambda code, p, message: ctx.issue(issue_codes.get(code, 'production_' + code), p, message),
        gap=lambda code, p, message: ctx.gap(gap_codes.get(code, 'production_' + code), p, message),
    ), effect, path)
    if view is None:
        return None
    if view['kind'] == 'access':
        if view['mode'] is AccessMode.READ:
            ctx.issue('production_read_only', path, 'READ-only access cannot be a write production')
        before = view['before']
        if before is not None and type(before.value) is not Unresolved:
            ctx.check_support((before.value,), OccurrencePoint(effect.operation, OccurrencePhase.BEFORE), path)
        return {'reference': effect, 'operation': view['operation'], 'access': view['access'],
                'before': before, 'after': view['after']}

    before = view['input']
    if before is not None and type(before.value) is not Unresolved:
        ctx.check_support((before.value,), OccurrencePoint(effect.invocation, OccurrencePhase.BEFORE), path)
    ctx.gap('production_invoke_write_unverified', path,
            'state transport alone does not prove native write semantics/bytes')
    return {'reference': effect, 'operation': view['operation'], 'input': before,
            'after': view['formal'], 'transfer': view['actual_transfer']}


def write_errors(ctx):
    from scar.ir.execution.source_validation import resolve_edge, resolve_endpoint, resolve_raw

    ctx.resolved_writes = {}
    write_relations = {EvidenceRelation.PRODUCES, EvidenceRelation.OBSERVED_WRITE,
                       EvidenceRelation.OVERWRITES, EvidenceRelation.MATERIALIZES}
    for row in ctx.writes.values():
        path = f'writes[{row.id}]'
        start = len(ctx.gaps)
        writer, value, effect, operation, run = None, None, None, None, None
        if type(row.writer) is WorkWrite:
            writer = ctx.submissions.get(row.writer.ticket)
            if writer is None:
                ctx.issue('production_ticket', path, 'writer references an absent work ticket')
            elif type(writer.origin) is WithinOperation:
                operation = writer.origin.operation
                run = ctx.processes[ctx.activations[ctx.activation_of(operation)].process].run
            else:
                run = writer.origin.point.run
        elif type(row.writer) is HostWrite:
            operation = row.writer.operation
            writer = ctx.occurrences.get(operation)
            if writer is None:
                ctx.issue('production_host', path, 'host writer references an absent operation occurrence')
                operation = None
            else:
                run = ctx.processes[ctx.activations[ctx.activation_of(operation)].process].run
        elif type(row.writer) is ExternalEstablishment:
            writer = row.writer
            ctx.check_point(writer.point, path)
            run = writer.point.run
            ctx.gap('production_external_establishment_unverified', path, 'external establishment has no native/content correspondence proof')
        else:
            ctx.check_point(row.writer.point, path)
            ctx.gap('production_writer_unresolved', path, row.writer.reason)
        if type(row.production) is ValueProduction:
            value = ctx.values.get(row.production.value)
            if value is None:
                ctx.issue('production_value', path, 'production references an absent original value occurrence')
            else:
                definition = ctx.core_values[value.value]
                if type(definition) not in (DataValue, ReferenceValue):
                    ctx.issue('production_value_kind', path, 'state/binding/protocol is not a logical value production')
                value_run = ctx.processes[ctx.activations[ctx.value_activation(value.id)].process].run
                if run is not None and run != value_run:
                    ctx.issue('production_run', path, 'writer and value production belong to different runs')
                if operation is not None and type(value.origin) is OperationOccurrenceID and value.origin != operation:
                    if ctx.activation_of(value.origin) == ctx.activation_of(operation):
                        ctx.issue('production_value_owner', path, 'another sequential operation produced this value')
                    else:
                        ctx.gap('production_nested_value_unverified', path, 'nested result-to-writer pairing is not established')
        elif type(row.production) is StateProduction:
            effect = _effect(ctx, row.production.effect, path)
            if effect is not None:
                actual_operation = effect['operation'].id
                if operation is not None and actual_operation != operation:
                    if ctx.activation_of(actual_operation) == ctx.activation_of(operation):
                        ctx.issue('production_effect_owner', path, 'effect belongs to another sequential writer occurrence')
                    else:
                        ctx.gap('production_nested_effect_unverified', path, 'nested effect-to-writer pairing is not established')
                effect_run = ctx.processes[ctx.activations[ctx.activation_of(actual_operation)].process].run
                if run is not None and effect_run != run:
                    ctx.issue('production_run', path, 'writer and effect belong to different runs')
                if type(row.writer) is WorkWrite and writer is not None and row.production.effect not in writer.effects:
                    if writer.effects_coverage is Completeness.COMPLETE:
                        ctx.issue('production_submitted_effect', path, 'write contradicts the complete submitted effect association')
                    else:
                        ctx.gap('production_submitted_effect_unresolved', path, 'exact effect has no submitted work association')
        else:
            ctx.gap('production_target_unresolved', path, row.production.reason)
        stage = None
        if type(row.stage) is RawRecordRef:
            stage = resolve_raw(ctx, row.stage, path)
        elif type(row.stage) is SourceEdgeRef:
            stage = resolve_edge(ctx, row.stage, path)
            if stage is not None and stage['edge'].relation not in write_relations:
                ctx.issue('production_source_relation', path, 'stage relation is not a write/production relation')
        elif type(row.stage) is NativeOutputRef:
            stage = native_port(ctx, row.stage, path)
            if stage is not None and value is not None and stage['value'] is not None and stage['value'].id != value.id:
                ctx.issue('production_native_actual', path, 'native stage selects another actual output incarnation')
        else:
            ctx.gap('production_stage_unresolved', path, row.stage.reason)
        sources, source_pairs = [], []
        for reference in row.evidence:
            resolved = resolve_edge(ctx, reference, path)
            if resolved is not None:
                sources.append(resolved)
                source_pairs.append((reference, resolved))
                if resolved['edge'].relation not in write_relations:
                    ctx.issue('production_source_relation', path, 'write evidence must retain a write/production relation')
        for reference in (row.stage, *row.evidence):
            if type(reference) in (RawRecordRef, SourceEdgeRef):
                inventory = ctx.source_inventories.get(reference.source.inventory)
                if inventory is not None and run is not None and inventory.run != run:
                    ctx.issue('production_source_run', path, 'source inventory and writer belong to different runs')
        placement = resolve_placement(ctx, row.placement, path)
        endpoint_profile, endpoint_request, byte_input = None, None, None
        if type(row.placement) is EndpointContent:
            if type(row.placement.selector) is ProtocolSelector:
                endpoint_profile = resolve_endpoint_profile(ctx, row.placement.selector, path)
                if endpoint_profile is not None:
                    if endpoint_profile['operation'] != 'WRITE_REQUEST':
                        ctx.issue('production_endpoint_profile_role', path,
                                  'endpoint write placement requires a WRITE_REQUEST profile')
                    endpoint_request = endpoint_extent(ctx, row.placement.target, endpoint_profile, path,
                        expected_operation='WRITE_REQUEST',
                        materialization=row.placement.materialization)
                    byte_input = source_byte_input(ctx, row, {'writer': writer}, path)
                    if (byte_input is not None and endpoint_request is not None
                            and endpoint_request['length_bytes'] != byte_input['extent']['length_bytes']):
                        ctx.issue('production_endpoint_request_length', path,
                                  'write request byte length differs from the unique original source ContentRead geometry')
            else:
                ctx.gap('production_endpoint_profile_unresolved', path,
                        'endpoint write requires an explicit WRITE_REQUEST ProtocolSelector')
        if (type(row.writer) is WorkWrite and writer is not None
                and type(row.placement) is not Unresolved):
            from scar.ir.execution.content_validation import check_work_materialization
            check_work_materialization(ctx, writer, row.placement.materialization, path)
        edge_sources = ([(row.stage, stage)] if type(row.stage) is SourceEdgeRef and stage is not None else []) + source_pairs
        for reference, resolved in edge_sources:
            edge = resolved['edge']
            endpoint = resolve_endpoint(ctx, SourceEndpointRef(reference.source, edge.target), path)
            resolved['target'] = endpoint
            if (endpoint is not None and placement is not None
                    and artifact_key(endpoint['values_artifact']) == artifact_key(row.placement.materialization.artifact)):
                materialization = placement[0]
                if (endpoint['value_version'] is not None
                        and endpoint['value_version'] != materialization.value_version):
                    ctx.issue('production_source_version', path, 'original same-bundle source endpoint names another logical version')
                if (endpoint['materialization'] is not None
                        and endpoint['materialization'] != materialization.id):
                    ctx.issue('production_source_materialization', path, 'original same-bundle source endpoint names another materialization')
        ctx.gap('production_source_join_unverified', path, 'source stage/native write-to-effect/placement correspondence is not established by labels or shared host ownership')
        ctx.gap('production_state_content_unverified', path, 'original state/value and source endpoints do not prove physical content/version production')
        ctx.resolved_writes[row.id] = {'row': row, 'writer': writer, 'production': row.production,
            'value': value, 'effect': effect, 'placement': placement, 'stage': stage,
            'sources': tuple(sources), 'gaps': tuple(ctx.gaps[start:]),
            'endpoint_profile': endpoint_profile, 'endpoint_request': endpoint_request,
            'source_byte_input': byte_input}


def produced_errors(ctx):
    from .production_ranges import resolve_actual_range

    ctx.resolved_produced = {}
    for row in ctx.produced.values():
        path = f'produced[{row.id}]'
        start = len(ctx.gaps)
        write = ctx.writes.get(row.write)
        outcome = ctx.outcomes.get(row.outcome)
        if write is None:
            ctx.issue('production_write', path, 'produced association references an absent write')
        if outcome is None:
            ctx.issue('production_outcome', path, 'produced association references an absent actual outcome')
        if write is None or outcome is None:
            continue
        if type(write.writer) is not WorkWrite or write.writer.ticket != outcome.ticket:
            ctx.issue('production_outcome_ticket', path, 'resource outcome must belong to this exact work writer')
        frontier = None
        if type(write.production) is ValueProduction:
            if row.frontier is not None:
                ctx.issue('production_value_frontier', path, 'value production has no state frontier: use None/NotApplicable')
        elif type(write.production) is StateProduction:
            if row.frontier is None:
                ctx.issue('production_state_frontier', path, 'state production requires an actual frontier index or explicit unresolved')
            elif type(row.frontier) is int:
                if row.frontier >= len(outcome.effect_frontier):
                    ctx.issue('production_frontier', path, 'frontier index does not select an actual outcome effect')
                else:
                    frontier = outcome.effect_frontier[row.frontier]
                    if frontier != write.production.effect:
                        ctx.issue('production_frontier_effect', path, 'frontier selects another original effect')
            else:
                ctx.gap('production_frontier_unresolved', path, row.frontier.reason)
        else:
            ctx.gap('production_frontier_unresolved', path, 'production target is unresolved')
        ranges = resolve_actual_range(ctx, write, row.actual, row.rule,
                                      outcome.protocol_result, path)
        actual, requested, profile = (ranges[key] for key in ('actual', 'requested', 'rawprofile'))
        ctx.gap('production_native_protocol_unverified', path, 'parsed declaration does not establish physical write/native applicability, source authority or content stability')
        ctx.resolved_produced[row.id] = {'row': row, 'write': write, 'outcome': outcome,
            'frontier': frontier, 'protocol_result': outcome.protocol_result, 'rawprofile': profile,
            'actual_selector': ranges['actual_selector'],
            'placement_match': ranges['placement_match'],
            'actual': actual, 'requested': requested, 'conditions': ('native/source applicability remains unverified',),
            'gaps': tuple(ctx.gaps[start:])}


# Shared original-result/actual-selector guard for work and host production.
#
# The fixed byte-prefix profile is a declaration parser, not a native adapter.
# It preserves P diagnostics and original artifact-scoped regions verbatim.
def _profile(ctx, resolved, path):
    record = resolved['record']
    payload = record.payload
    if type(payload) is not dict:
        ctx.gap('production_profile_unavailable', path, 'raw record has no decoded JSON object payload')
        return None
    metadata = payload.get('metadata')
    keys = {'schema', 'schema_version', 'stage', 'requested_bytes', 'written_bytes'}
    if type(metadata) is not dict or 'scar_native_result' not in metadata:
        ctx.gap('production_profile_unavailable', path, 'exact raw record has no byte-prefix protocol payload')
        return None
    profile = metadata['scar_native_result']
    if (type(profile) is not dict or set(profile) != keys
            or profile['schema'] != 'scar.native.byte-prefix-write'
            or type(profile['schema_version']) is not int or profile['schema_version'] != 1
            or type(profile['stage']) is not str or not profile['stage']):
        ctx.issue('production_profile', path, 'native result must have the exact closed byte-prefix schema')
        return None
    requested, written = profile['requested_bytes'], profile['written_bytes']
    if (type(requested) is not int or type(written) is not int
            or requested < 0 or written < 0 or written > requested):
        ctx.issue('production_profile_count', path, 'native byte counts must be exact integers with 0 <= written <= requested')
        return None
    return profile


def _actual_placement(ctx, write, actual_ref, actual, path):
    """Check physical correspondence independently of a result protocol.

    Unknown write semantics cannot waive a known storage/Mat contradiction.
    Endpoint selectors need their own interpreted placement protocol.
    """
    if type(actual_ref) is not RegionRef or actual is None:
        return None
    resolved = ctx.resolved_writes.get(write.id)
    if (type(write.placement) is not StorageContent or resolved is None
            or resolved['placement'] is None):
        ctx.gap('production_actual_placement_unresolved', path,
                'actual byte region has no interpreted original storage materialization placement')
        return None
    mat, _ = resolved['placement']
    artifact = write.placement.materialization.artifact
    if artifact_key(actual_ref.artifact) != artifact_key(artifact):
        ctx.issue('production_selector_scope', path,
                  'actual and original materialization belong to different exact ValueGraphs')
        return False
    if mat.region is None:
        ctx.gap('production_actual_placement_unresolved', path,
                'original materialization has no storage region for this actual byte selector')
        return None
    graph = ctx.external_values(artifact, path)
    if graph is None:
        return None
    region = graph.regions[mat.region]
    if actual.allocation != region.allocation:
        ctx.issue('production_selector_allocation', path,
                  'actual selector belongs to another allocation incarnation')
        return False
    original = RegionRef(artifact, mat.region)
    coverage = _storage_coverage(actual_ref, (original,),
        catalog=_ValueCatalog.from_snapshots(ctx.snapshots), budget=4096)
    if coverage.status is CoverageStatus.INCOMPLETE:
        ctx.issue('production_actual_outside', path,
                  'actual selector extends outside the original materialization region')
        return False
    if coverage.status is CoverageStatus.UNRESOLVED:
        ctx.gap('production_actual_coverage_unresolved', path,
                'actual selector subset is not established within geometry budget')
        return None
    return True


def resolve_actual_range(ctx, write, actual_ref, rule, result, path):
    from scar.ir.execution.source_validation import resolve_raw

    actual = None
    actual_endpoint = None
    actual_profile = None
    if type(actual_ref) is RegionRef:
        graph = ctx.external_values(actual_ref.artifact, path)
        actual = None if graph is None else graph.regions.get(actual_ref.region)
        if graph is not None and actual is None:
            ctx.issue('production_actual_region', path, 'actual selector is absent from exact ValueGraph')
    elif type(actual_ref) is ProtocolSelector:
        if type(write.placement) is not EndpointContent:
            ctx.issue('production_actual_endpoint_placement', path,
                      'endpoint byte actual requires the original endpoint content placement')
        else:
            actual_profile = resolve_endpoint_profile(ctx, actual_ref, path)
            if actual_profile is not None:
                if actual_profile['operation'] != 'WRITE_ACTUAL':
                    ctx.issue('production_actual_endpoint_profile', path,
                              'actual endpoint selector requires a WRITE_ACTUAL profile')
                actual_endpoint = endpoint_extent(ctx, write.placement.target, actual_profile, path,
                    expected_operation='WRITE_ACTUAL',
                    materialization=write.placement.materialization)
    else:
        ctx.gap('production_actual_unresolved', path, 'actual selector has no parsed byte geometry')
    placement_match = _actual_placement(ctx, write, actual_ref, actual, path)
    raw, profile, requested = None, None, None
    actual_selector = actual_ref if type(actual_ref) is RegionRef else actual_endpoint
    if type(rule) is BytePrefixWrite:
        if type(write.stage) is not RawRecordRef:
            ctx.gap('production_profile_stage_unresolved', path, 'byte-prefix profile requires an exact original raw record stage')
        else:
            raw = resolve_raw(ctx, write.stage, path)
            if raw is not None:
                profile = _profile(ctx, raw, path)
        resolved = ctx.resolved_writes.get(write.id)
        if type(write.placement) is not StorageContent:
            if type(write.placement) is not EndpointContent:
                ctx.gap('production_profile_placement_unresolved', path,
                        'byte-prefix rule requires an interpreted storage or endpoint placement')
            elif type(write.placement.selector) is not ProtocolSelector:
                ctx.gap('production_endpoint_request_unresolved', path,
                        'endpoint placement has no original WRITE_REQUEST profile')
            else:
                resolved = ctx.resolved_writes.get(write.id)
                requested = None if resolved is None else resolved.get('endpoint_request')
                if requested is None:
                    ctx.gap('production_endpoint_request_unresolved', path,
                            'original endpoint request range was not interpreted')
                if actual_endpoint is None:
                    ctx.gap('production_endpoint_actual_unresolved', path,
                            'actual endpoint range was not interpreted')
                if requested is not None and actual_endpoint is not None:
                    if (requested['operation'] != 'WRITE_REQUEST'
                            or requested['profile'] != actual_endpoint['profile']):
                        ctx.issue('production_endpoint_profile_pair', path,
                                  'request and actual must use this exact fixed byte-prefix interpretation')
                    if requested['offset_bytes'] != actual_endpoint['offset_bytes']:
                        ctx.issue('production_endpoint_offset', path,
                                  'actual endpoint prefix must use the exact request offset')
                    if profile is not None and requested['length_bytes'] != profile['requested_bytes']:
                        ctx.issue('production_endpoint_request_count', path,
                                  'request profile length differs from the exact raw requested byte count')
                    if profile is not None and actual_endpoint['length_bytes'] != profile['written_bytes']:
                        ctx.issue('production_endpoint_actual_count', path,
                                  'actual profile length differs from the exact raw written byte count')
                    if (target_key(requested['target']) != target_key(actual_endpoint['target'])
                            or not same_materialization(requested['materialization'],
                                                       actual_endpoint['materialization'])):
                        ctx.issue('production_endpoint_actual_placement', path,
                                  'actual range must keep the original endpoint instance and exact destination MatRef')
                    placement_match = bool(
                        profile is not None and actual_profile is not None
                        and requested['length_bytes'] == profile['requested_bytes']
                        and actual_endpoint['length_bytes'] == profile['written_bytes']
                        and actual_profile['operation'] == 'WRITE_ACTUAL'
                        and requested['offset_bytes'] == actual_endpoint['offset_bytes'] == 0
                        and target_key(requested['target']) == target_key(actual_endpoint['target'])
                        and same_materialization(requested['materialization'],
                                                 actual_endpoint['materialization']))
        elif resolved is not None and resolved['placement'] is not None:
            mat, selector = resolved['placement']
            if selector is not None:
                graph = ctx.external_values(selector.artifact, path)
                requested = None if graph is None else graph.regions.get(selector.region)
            if requested is not None and actual is not None:
                if requested.allocation != actual.allocation:
                    ctx.issue('production_selector_allocation', path, 'actual selector belongs to another allocation incarnation')
                linear = lambda x: x.dtype == 'uint8' and x.layout == 'strided' and len(x.shape) == 1 and x.strides == (1,)
                if not linear(requested) or not linear(actual):
                    ctx.gap('production_byte_layout_unresolved', path, 'prefix rule only interprets linear rank1 uint8 byte buffers')
                elif profile is not None:
                    if requested.shape != (profile['requested_bytes'],) or actual.shape != (profile['written_bytes'],):
                        ctx.issue('production_selector_count', path, 'original selector geometry does not equal parsed requested/produced byte counts')
                    if actual.offset != requested.offset:
                        ctx.issue('production_selector_offset', path, 'actual byte prefix must start at the original request offset')
        if type(result) is ProviderFactReference and raw is not None:
            original = raw['source'].raw_artifact
            if (artifact_key(result.artifact) != artifact_key(original)
                    or result.locator != f'#L{write.stage.line}/metadata/scar_native_result'):
                ctx.issue('production_protocol_result', path, 'protocol result must reference this exact raw record payload')
        elif type(result) is not Unresolved and result in ctx.values:
            definition = ctx.core_values[ctx.values[result].value]
            origin = definition.origin
            integer = (origin.value if type(origin) is LiteralOrigin and type(origin.value) is int
                       else origin.literal.to_python() if type(origin) is ExactLiteralOrigin
                       and origin.literal.kind is LiteralKind.INT else None)
            if isinstance(definition, DataValue) and integer is not None and profile is not None:
                if integer != profile['written_bytes']:
                    ctx.issue('production_protocol_result', path, 'literal protocol result conflicts with parsed produced bytes')
            else:
                ctx.gap('production_result_unresolved', path, 'original result value has no decoded exact integer payload/port correspondence')
        else:
            ctx.gap('production_result_unresolved', path, 'protocol result has no parsed exact raw/result correspondence')
    else:
        ctx.gap('production_rule_unresolved', path, rule.reason)
    return {'actual': actual_endpoint if actual_endpoint is not None else actual,
            'actual_selector': actual_selector, 'requested': requested, 'rawprofile': profile,
            'raw': raw, 'protocol_result': result, 'placement_match': placement_match}


__all__ = [
    'production',
    'production_shared',
    'write_errors',
    'produced_errors',
    'resolve_actual_range',
]

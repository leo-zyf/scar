"""Source validation."""

from scar.ir.v2.evidence import (
    EvidenceEndpoint,
    EvidenceGraph,
    EvidenceNodeKind as N,
    EvidenceNodeKind,
)
from scar.ir.execution.content import ObjectRef, VersionRef, ObjectIdentity
from scar.ir.execution.content_validation import artifact_key
from scar.ir.execution.coverage import RegionRef
from scar.ir.execution.inventory import InventoryResolver
from scar.ir.execution.production import RawRecordRef, SourceEdgeRef, SourceEndpointRef
from scar.ir.execution.resources import AllocationTarget
from scar.ir.execution.snapshot import SnapshotError
from scar.ir.execution.work import MaterializationRef
from scar.ir.program.model import ReferenceValue
from scar.ir.v2.values import ValueGraph
from scar.ir.execution.common import Unresolved
from scar.ir.execution.coordinates import ValueOccurrence
from scar.ir.execution.validity_budget import ValidityBudget


# Resolve independent source inventories and scoped endpoint associations.
#
# Format replay gives structural correspondence inside one bundle. It does not
# certify capture completeness, physical writes or cross-artifact value equality.
def source_inventory_errors(ctx):
    """Load sources before resource/D validation so bundle scope is reusable."""
    ctx.source_resolutions, ctx.inventory_sources = {}, {}
    replayed, included = {}, set()
    for binding in ctx.source_inventories.values():
        path = f'source_inventories[{binding.id}]'
        if binding.run not in ctx.runs:
            ctx.issue('inventory_run', path, 'source inventory references an absent canonical run')
        try:
            key = artifact_key(binding.manifest), artifact_key(binding.index)
            if key not in replayed:
                replayed[key] = InventoryResolver(ctx.snapshots).resolve(binding.manifest, binding.index)
            result = replayed[key]
            ctx.source_resolutions[binding.id] = result
            for gap in result.gaps:
                ctx.gap(gap.code, f'{path}.{gap.path}', gap.message)
            for source in result.sources:
                ctx.inventory_sources[binding.id, source.key] = source
                if source.bundle is not None:
                    for artifact, graph in (
                        (source.evidence_artifact, source.bundle.evidence),
                        (source.values_artifact, source.bundle.values),
                        (source.semantic_artifact, source.bundle.semantic),
                    ):
                        component = artifact_key(artifact)
                        if component not in included:
                            ctx.snapshots.include(artifact, graph)
                            included.add(component)
        except SnapshotError as error:
            ctx.issue(error.code, path, str(error))
        except (TypeError, ValueError, KeyError, AttributeError, OverflowError, RecursionError) as error:
            ctx.issue('inventory_invalid', path, str(error))


def _source(ctx, reference, path):
    binding = ctx.source_inventories.get(reference.inventory)
    if binding is None:
        ctx.issue('inventory_binding', path, 'source refers to an absent inventory binding')
        return None
    source = ctx.inventory_sources.get((reference.inventory, reference.stream))
    if source is None:
        resolution = ctx.source_resolutions.get(reference.inventory)
        known_empty = resolution is not None and any(
            row.code == 'inventory_sources_empty' for row in resolution.gaps)
        if resolution is None or not resolution.sources and not known_empty:
            ctx.gap('inventory_source_pending', path, 'source manifest was not resolved; source membership remains pending')
        else:
            ctx.issue('inventory_stream', path, 'source stream is absent from its exact manifest')
        return None
    return {'source': source, 'inventory': binding, 'run': binding.run,
            'raw_artifact': source.raw_artifact, 'artifact': source.evidence_artifact,
            'graph': None if source.bundle is None else source.bundle.evidence}


def resolve_raw(ctx, reference, path):
    if type(reference) is not RawRecordRef:
        raise TypeError('resolve_raw requires exact RawRecordRef')
    reference.__post_init__()
    result = _source(ctx, reference.source, path)
    if result is None:
        return None
    source = result['source']
    record = next((row for row in source.raw_records if row.line == reference.line), None)
    if record is None:
        if source.projection_replayed:
            ctx.issue('inventory_raw_line', path, 'raw line is absent from exact source bytes')
        else:
            ctx.gap('inventory_raw_pending', path, 'raw line/projection is pending, not an absent write')
        return None
    if record.sha256 != reference.sha256.lower():
        ctx.issue('inventory_raw_digest', path, 'raw reference names different physical line bytes')
        return None
    if record.payload is None:
        ctx.gap('inventory_raw_unparsed', path, 'physical source line has no parsed JSON object')
        return None
    return {**result, 'record': record}


def resolve_edge(ctx, reference, path):
    if type(reference) is not SourceEdgeRef:
        raise TypeError('resolve_edge requires exact SourceEdgeRef')
    reference.__post_init__()
    result = _source(ctx, reference.source, path)
    if result is None:
        return None
    if result['graph'] is None:
        ctx.gap('inventory_projection_pending', path, 'source projection is unavailable; source edges remain pending')
        return None
    edge = result['graph'].edges.get(reference.edge)
    if edge is None:
        ctx.issue('inventory_edge', path, 'source edge is absent from its exact normalized projection')
        return None
    return {**result, 'edge': edge}


def resolve_endpoint(ctx, reference, path):
    """Derive facets only within the strict replayed IRBundle component scope."""
    if type(reference) is not SourceEndpointRef:
        raise TypeError('resolve_endpoint requires exact SourceEndpointRef')
    reference.__post_init__()
    result = _source(ctx, reference.source, path)
    if result is None:
        return None
    source, endpoint = result['source'], reference.endpoint
    if result['graph'] is None:
        ctx.gap('inventory_projection_pending', path, 'source endpoint projection is pending')
        return None
    if not result['graph']._endpoint_exists(endpoint):
        ctx.issue('inventory_endpoint', path, 'endpoint is absent from its exact normalized projection')
        return None
    graph = source.bundle.values
    version = mat = obj = region = allocation = observation = None
    if endpoint.kind is N.VALUE_OBSERVATION:
        observation = result['graph'].observations[endpoint.reference]
        version, mat = observation.version, observation.materialization
    elif endpoint.kind is N.VALUE_VERSION:
        version = endpoint.reference
    elif endpoint.kind is N.MATERIALIZATION:
        mat = endpoint.reference
    elif endpoint.kind is N.OBJECT:
        obj = endpoint.reference
        versions = {row.value_version for row in graph.bindings if row.object_id == obj}
        if len(versions) == 1:
            version = next(iter(versions))
        elif len(versions) > 1:
            ctx.gap('inventory_object_versions', path, 'object bindings span multiple versions; no current version is selected')
    elif endpoint.kind is N.STORAGE_REGION:
        region = endpoint.reference
    elif endpoint.kind is N.ALLOCATION:
        allocation = endpoint.reference
    if mat is not None:
        materialization = graph.materializations.get(mat)
        if materialization is None:
            ctx.issue('inventory_materialization', path, 'bundle endpoint materialization is absent from its original ValueGraph')
        else:
            version, region = materialization.value_version, materialization.region
    if region is not None:
        original = graph.regions.get(region)
        if original is None:
            ctx.issue('inventory_region', path, 'bundle endpoint region is absent from its original ValueGraph')
        else:
            allocation = original.allocation
    if version is not None and version not in graph.versions:
        ctx.issue('inventory_version', path, 'bundle endpoint version is absent from its original ValueGraph')
    if allocation is not None and allocation not in graph.allocations:
        ctx.issue('inventory_allocation', path, 'bundle endpoint allocation is absent from its original ValueGraph')
    return {**result, 'endpoint': endpoint, 'observation': observation,
            'value_version': version, 'materialization': mat, 'object': obj,
            'region': region, 'allocation': allocation,
            'values_artifact': source.values_artifact}


def target_original(ctx, target, path):
    """Return a scoped existing target facet, never a synthesized producer."""
    if type(target) is ObjectRef and target.artifact.schema == EvidenceGraph.SCHEMA:
        graph = ctx._external(target.artifact, path, EvidenceGraph, ctx.evidence_graphs)
        if graph is None:
            return None
        endpoint = EvidenceEndpoint(N.OBJECT, target.object)
        if not graph._endpoint_exists(endpoint):
            ctx.issue('inventory_bridge_target', path, 'object target is absent from its exact EvidenceGraph registry')
            return None
        return {'facet': 'object', 'identity': target.object, 'original': endpoint,
                'graph': graph, 'artifact': target.artifact}
    graph = ctx.external_values(target.artifact, path)
    if graph is None:
        return None
    if type(target) is VersionRef:
        facet, identity, original = 'value_version', target.version, graph.versions.get(target.version)
    elif type(target) is MaterializationRef:
        facet, identity = 'materialization', target.materialization
        original = graph.materializations.get(identity)
    elif type(target) is ObjectRef:
        facet, identity = 'object', target.object
        original = graph.objects.get(identity)
        if original is None:
            original = tuple(row for row in graph.bindings if row.object_id == identity) or None
    elif type(target) is RegionRef:
        facet, identity, original = 'region', target.region, graph.regions.get(target.region)
    elif type(target) is AllocationTarget:
        facet, identity = 'allocation', target.allocation
        original = graph.allocations.get(identity)
    else:
        raise TypeError('source bridge target must be a scoped ValueGraph facet')
    if original is None:
        ctx.issue('inventory_bridge_target', path, 'bridge target is absent from its exact original ValueGraph')
        return None
    return {'facet': facet, 'identity': identity, 'original': original, 'graph': graph,
            'artifact': target.artifact}


def source_bridge_errors(ctx):
    ctx.resolved_source_bridges = {}
    for row in ctx.source_bridges.values():
        path = f'source_bridges[{row.id}]'
        source = resolve_endpoint(ctx, row.endpoint, path)
        target = target_original(ctx, row.target, path)
        if source is not None and target is not None:
            source_artifact = (source['artifact'] if target['facet'] == 'object'
                               and target['artifact'].schema == EvidenceGraph.SCHEMA
                               else source['values_artifact'])
            same_scope = artifact_key(source_artifact) == artifact_key(target['artifact'])
            original = source[target['facet']]
            if same_scope and original is not None and original != target['identity']:
                ctx.issue('inventory_bridge_conflict', path, 'bridge contradicts the original endpoint identity in the same exact ValueGraph')
            elif original is None:
                ctx.gap('inventory_bridge_facet_unresolved', path, 'source endpoint does not select this facet; an association is not derivation proof')
            if not same_scope:
                ctx.gap('inventory_bridge_cross_scope_unverified', path,
                        'explicit cross-artifact association has no content/native identity proof; bare IDs are not compared')
        ctx.gap('inventory_bridge_runtime_unverified', path, 'endpoint bridge is a declared association, not runtime/capture authority')
        ctx.resolved_source_bridges[row.id] = {'row': row, 'source': source, 'target': target}


# Local comparison of independently declared current Reference identities.
#
# This query creates no copy edge and says nothing about an object's fields
# between two reads. Equal scoped ObjectIDs are conditional declarations, not
# native identity, address-generation or physical-lifetime verification.
def _flags():
    return {'source_runtime_verified': False, 'native_identity_verified': False,
            'runtime_identity_verified': False, 'value_identity_verified': False,
            'transformation_authorized': False}


def _endpoint(ctx, role, value):
    if value is not None and type(value) not in (ValueOccurrence, Unresolved):
        raise TypeError('object_identity endpoints must be ValueOccurrence or unresolved')
    captured = (value if type(value) is ValueOccurrence
                and ctx.values.get(value.id) == value else None)
    resolved = None if captured is None else ctx.resolved_denotations.get(captured.id)
    denotation = None if resolved is None else resolved['denotation']
    return {'role': role, 'value': value,
            'definition': None if captured is None else ctx.core_values.get(captured.value),
            'denotation': denotation,
            'declared_basis': None if denotation is None else denotation.basis,
            'basis': None if resolved is None else resolved['basis'],
            'identity': None if denotation is None else denotation.identity,
            'process': None, 'run': None, 'registered': None,
            'captured': captured is not None, 'gaps': ()}


def object_identity(ctx, source, target, *, shared):
    """Compare two original endpoints under one shared, query-local budget.

    The complete two-endpoint inventory precedes budgeted joins. A missing
    source catalog remains unknown even when both denotations name the same
    ObjectID. No diagnostic is added to ``ctx`` for different candidate IDs.
    The public caller refreshes external snapshots and detaches the report.
    """
    if not isinstance(shared, ValidityBudget):
        raise TypeError('object_identity requires a shared ValidityBudget')
    inventory = (_endpoint(ctx, 'source', source), _endpoint(ctx, 'target', target))
    gaps, cursor = [], None

    def gap(code, endpoint=None):
        if code not in gaps:
            gaps.append(code)
        if endpoint is not None and code not in endpoint['gaps']:
            endpoint['gaps'] = (*endpoint['gaps'], code)

    def spend(phase, endpoint=None, *, cost=1):
        nonlocal cursor
        path = 'object_identity.' + (endpoint['role'] + '.' if endpoint else '') + phase
        if shared.truncated or not shared.spend('object_identity_' + phase, cost, path=path):
            if cursor is None:
                cursor = {'phase': phase, 'endpoint': None if endpoint is None else endpoint['role']}
            gap('object_identity_budget_exhausted', endpoint)
            return False
        return True

    for endpoint in inventory:
        if not spend('endpoints', endpoint):
            break
        if not endpoint['captured']:
            gap('object_identity_value_unobserved', endpoint)
            continue
        if type(endpoint['definition']) is not ReferenceValue:
            gap('object_identity_reference_required', endpoint)
            continue
        if not spend('scopes', endpoint):
            break
        activation = ctx.activations.get(ctx.value_activation(endpoint['value'].id))
        process = None if activation is None else ctx.processes.get(activation.process)
        if process is None or process.run not in ctx.runs:
            gap('object_identity_execution_scope_unproved', endpoint)
        else:
            endpoint['process'], endpoint['run'] = process.id, process.run
        if not spend('denotations', endpoint):
            break
        if endpoint['denotation'] is None:
            gap('object_identity_denotation_missing', endpoint)
            continue
        identity = endpoint['identity']
        if type(identity) is not ObjectIdentity:
            gap('object_identity_identity_unresolved', endpoint)
            continue
        if type(endpoint['declared_basis']) is Unresolved or endpoint['basis'] is None:
            # The declaration remains a conditional model fact. Its missing
            # source/native correspondence is retained, never promoted.
            gap('object_identity_basis_unresolved', endpoint)
        if not spend('registries', endpoint):
            break
        reference = identity.object
        family = (EvidenceGraph if reference.artifact.schema == EvidenceGraph.SCHEMA
                  else ValueGraph if reference.artifact.schema == ValueGraph.SCHEMA else None)
        if family is None:
            gap('object_identity_artifact_scope_unproved', endpoint)
            continue
        try:
            graph = ctx.snapshots.graph(reference.artifact, family)
        except SnapshotError as error:
            endpoint['catalog_error'] = {'code': error.code, 'message': str(error)}
            gap('object_identity_catalog_invalid', endpoint)
            continue
        if graph is None:
            gap('object_identity_catalog_unavailable', endpoint)
            continue
        if family is EvidenceGraph:
            endpoint['registered'] = (EvidenceNodeKind.OBJECT, reference.object.wire) in graph.external_references
        else:
            registered = False
            if graph.objects:
                if spend('registry_objects', endpoint):
                    registered = reference.object in graph.objects
                else:
                    registered = None
            if registered is False:
                for binding in graph.bindings:
                    if not spend('registry_bindings', endpoint):
                        registered = None
                        break
                    if binding.object_id == reference.object:
                        registered = True
                        break
            endpoint['registered'] = registered
        if endpoint['registered'] is False:
            gap('object_identity_object_unregistered', endpoint)
        if shared.truncated:
            break

    # Keep only the two endpoints' original source diagnostics. Repeating all
    # unrelated model gaps in every candidate would inflate detached reports.
    # This actual scan is charged, including paths which do not match.
    prefixes = tuple(f'denotations[{row["denotation"].id}]'
                     for row in inventory if row['denotation'] is not None)
    source_gaps = ()
    if spend('source_gap_rows', cost=len(ctx.gaps)):
        source_gaps = tuple(row for row in ctx.gaps
                            if any(row.path.startswith(prefix) for prefix in prefixes))

    match, status = None, 'UNPROVED'
    left, right = inventory
    ready = all(type(row['definition']) is ReferenceValue
                and type(row['identity']) is ObjectIdentity
                and row['captured'] and row['process'] is not None
                and row['run'] is not None and row['registered'] is True
                for row in inventory)
    if ready and spend('comparisons'):
        if (left['process'], left['run']) != (right['process'], right['run']):
            gap('object_identity_execution_scope_unproved')
        elif artifact_key(left['identity'].object.artifact) != artifact_key(right['identity'].object.artifact):
            gap('object_identity_artifact_scope_unproved')
        elif left['identity'].object.object == right['identity'].object.object:
            match, status = True, 'DECLARED_SAME_OBJECT'
        else:
            match, status = False, 'DECLARED_DISTINCT_OBJECTS'
    if shared.truncated:
        match, status = None, 'UNPROVED'
        gap('object_identity_budget_exhausted')
        if cursor is None:
            cursor = {'phase': 'comparison', 'endpoint': None}
    return {'source': source, 'target': target, 'match': match, 'status': status,
            'inventory': inventory, 'gaps': tuple(gaps), 'source_gaps': source_gaps,
            'cursor': cursor,
            'budget': {'limit': shared.limit, 'used': shared.used,
                       'costs': dict(shared.costs), 'truncated': shared.truncated},
            'conditions': ('scoped ObjectIdentity declarations remain conditional model facts',
                           'same object does not establish original copy or stored-field preservation',
                           'ObjectRef has no epoch; bare addresses, PIDs and URIs do not supply one'),
            **_flags()}


__all__ = [
    'source_inventory_errors',
    'source_bridge_errors',
    'resolve_raw',
    'resolve_edge',
    'resolve_endpoint',
    'target_original',
    'object_identity',
]

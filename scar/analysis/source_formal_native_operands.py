"""Join observed F4 formal references to exact native input observations."""

from collections import Counter, defaultdict
from pathlib import Path

from scar.analysis.source_observed_formal_instances import derive_observed_formal_instances
from scar.ir.program.call_edges import CapturedSite, InvocationKey
from scar.ir.record_codec import decode, encode, loads
from scar.trace.aten_scoped_ports import _capture_member
from scar.trace.capture_coordinates import bind_capture_coordinates
from scar.trace.dispatch_import import import_dispatch_bridge
from scar.trace.native_call_scope_import import _rf_member, import_native_call_scopes
from scar.trace.native_import import import_native_evidence
from scar.trace.tensor_physical_import import import_tensor_physical


def _replay(capture, max_bytes, max_lines):
    coordinates = bind_capture_coordinates(capture, max_bytes=max_bytes, max_lines=max_lines)

    def member(schema):
        return _capture_member(coordinates, schema)

    _, manifest = member('scar.execution-evidence-collection')
    eto, eto_path = member('pytorch.execution-trace.v1')
    kineto, kineto_path = member('pytorch.kineto.chrome-trace.v1')
    revision = loads(manifest.read_text())['provider']['torch_git_version']
    native = import_native_evidence(eto_path, kineto_path, max_bytes=max_bytes, provider_revision=revision)
    scopes = import_native_call_scopes(native, manifest, coordinates, max_bytes=max_bytes, max_lines=max_lines)
    import_dispatch_bridge(native, *(member(schema)[1] for schema in (
        'scar.dispatch-observation.events.v1', 'scar.dispatch-observation-trace',
        'scar.call-edge-events.v1', 'scar.call-edge-codes.v1', 'scar.call-edge-meta.v1')),
        native_manifest_path=manifest, coordinates=coordinates, max_bytes=max_bytes, max_events=max_lines)
    physical = import_tensor_physical(native, member('scar.tensor-physical-observation.v1')[1],
        member('scar.tensor-physical-trace')[1], coordinates=coordinates, max_bytes=max_bytes, max_rows=max_lines)
    return coordinates, native, scopes, physical, eto, kineto


def _input(instance, ordinal):
    matches = [slot for slot in instance.metadata.get('tensor_slots', ())
               if slot['role'] == 'inputs' and slot['ordered_path'] == [ordinal]]
    return matches[0] if len(matches) == 1 else None


def _links(native, scopes, physical, formals, eto, kineto):
    bridge = native.report['dispatch_bridge']
    bundle = native.bundle
    roots, children, by_reference = defaultdict(list), defaultdict(list), defaultdict(list)
    nodes = {node.metadata['node_id']: node for node in bundle.evidence.instances.values()
             if node.metadata.get('provider') == 'ETO'}
    for node in nodes.values():
        children[node.parent].append(node)
    for scope in scopes['observations']:
        if scope['status'] == 'CLOSED_SCOPE_OBSERVED' and scope['outcome'] == 'C_RETURN':
            key = decode(InvocationKey, scope['invocation']), decode(CapturedSite, scope['site'])
            roots[key].extend((scope, op) for op in scope['operators'])
    for row in formals:
        if row.observed_reference.ref.startswith('weak:'):
            by_reference[row.call_scope.invocation, row.observed_reference.ref].append(row)
    observations = {row['marker_instance']: row for row in bridge['observations']}
    regions = {(row['marker_instance'], row['argument_ordinal']): row for row in physical['associations']}
    storage = {region.id.wire: region for region in bundle.values.regions.values()}
    frame_artifact = bridge['artifacts']['calls']
    frame_paths = {frame_artifact['uri'], bridge['retained_input_paths']['calls']}

    links, matched = [], set()
    rejected = Counter()
    for path in bridge['paths']:
        observation = observations[path['marker_instance']]
        context = observation['retained_ancestor']
        if path['phase'] != 'span' or context is None:
            continue
        edge = bundle.evidence.edges[path['edge_id']]
        obj = bundle.values.objects[edge.source.reference]
        key = InvocationKey(native.report['pid'], observation['python_ident'], context['invocation'])
        rows = by_reference.get((key, obj.metadata['weak_token']))
        if not rows:
            continue
        ordinal = path['provider_argument_ordinal']
        if path['original_payload_path'] != [0, ordinal]:
            rejected['POSITIONAL_TENSOR_ORDINAL_NOT_ESTABLISHED'] += 1
            continue
        if context.get('site') is None:
            rejected['DISPATCH_SOURCE_CALL_SITE_NOT_OBSERVED'] += 1
            continue
        site = decode(CapturedSite, context['site'])
        ancestry = [item['node_id'] for item in observation['actual_provider_ancestry']]
        candidates = [(scope, op) for scope, op in roots[key, site]
                      if op['provider_node_id'] in ancestry[1:]]
        if (site.code != context['code'] or site.offset != context['bytecode_offset']
                or len(candidates) != 1 or not observation['native_tid_verified']
                or observation['native_tid'] != scopes['native_tid']):
            rejected['SAME_FRAME_CALL_SCOPE_AND_NATIVE_ROOT_NOT_UNIQUE'] += 1
            continue
        scope, root = candidates[0]
        if (scope['owner']['artifact'] != frame_artifact or any(
                row.entry.path not in frame_paths or scope['owner']['locator'] != f'#/lines/{row.entry.line_number}'
                for row in rows)):
            raise ValueError('selected source frame entry and native CALL use different capture evidence')
        marker = nodes[ancestry[0]]
        nested = [node for node in children[marker.id]
                  if bundle.semantic.definitions[node.definition].metadata.get('op_schema') == root['schema']]
        child = _rf_member(nested[0], pid=key.process_id, native_tid=observation['native_tid'],
                           category='cpu_op', eto=eto, kineto=kineto) if len(nested) == 1 else None
        marker_slot, root_slot = _input(marker, ordinal), _input(nodes[root['provider_node_id']], ordinal)
        child_slot = _input(nested[0], ordinal) if child is not None else None
        if (not marker_slot or not root_slot or not child_slot
                or marker_slot['descriptor_id'] != root_slot['descriptor_id']
                or marker_slot['descriptor_id'] != child_slot['descriptor_id']
                or marker_slot['provider_reference'] != edge.target.reference):
            rejected['EXACT_NATIVE_TENSOR_OPERAND_NOT_CORROBORATED'] += 1
            continue
        association = regions.get((marker.id.wire, ordinal))
        if (association is None or association['object'] != obj.id.wire
                or association['provider_reference'] != marker_slot['provider_reference']):
            rejected['PHYSICAL_MARKER_OPERAND_NOT_ADMITTED'] += 1
            continue
        region = storage[association['storage_region']]
        allocation = bundle.values.allocations[region.allocation]
        for row in rows:
            matched.add(row)
            links.append(dict(formal=encode(row), call_attempt=scope['attempt'], call_scope=scope['declaration'],
                call_root=root, dispatch=dict(sequence=observation['sequence'], marker_instance=marker.id.wire,
                    source_line=observation['source_line'], marker_eto_node=marker.metadata['node_id'],
                    dispatch_reference_edge=path['edge_id']), native_operator=child, input_ordinal=ordinal,
                native_input_reference=child_slot['raw_reference'], object=obj.id.wire,
                region=region.id.wire, allocation=allocation.id.wire,
                provider_allocation_id=allocation.metadata['provider_allocation_id'],
                physical_reference=association['physical_reference'],
                geometry=dict(device=region.device, dtype=region.dtype, offset=region.offset,
                              shape=list(region.shape), strides=list(region.strides))))
    rejected['FORMAL_ENTRY_WITHOUT_QUALIFIED_OPERAND'] = len(formals) - len(matched)
    return links, matched, {key: count for key, count in rejected.items() if count}


def export_source_formal_native_operands(session, capture, *, unit=None,
                                         max_bytes=128_000_000, max_lines=400_000) -> dict:
    """Replay one capture in memory and return an occurrence-level join receipt."""
    capture = Path(capture).resolve(strict=True)
    trace = session.report.get('trace')
    if trace is None or Path(trace['path']) != capture / 'trace':
        raise ValueError('F4 source observations and native operands must originate in the same capture')
    formals = derive_observed_formal_instances(session, unit=unit)
    coordinates, native, scopes, physical, eto, kineto = _replay(capture, max_bytes, max_lines)
    bridge = native.report['dispatch_bridge']
    frame_artifact = bridge['artifacts']['calls']
    source = session._f4.providers['binding_states'].index.codes
    receipts = [receipt for receipt in source.trace.frame_scan.files
                if receipt.family == 'events' and receipt.process_id == native.report['pid']]
    if (len(receipts) != 1 or receipts[0].raw_sha256 != frame_artifact['sha256']
            or receipts[0].raw_bytes != frame_artifact['size_bytes']):
        raise ValueError('F4 frame reader and native operands use different original frame bytes')
    source.check()
    links, matched, rejected = _links(native, scopes, physical, formals, eto, kineto)
    return dict(schema='scar.source-formal-native-operands', schema_version=1, capture=encode(coordinates.artifact),
        dispatch_artifact=bridge['artifacts']['observations'],
        physical_artifact=physical['artifacts']['observations'], links=links, rejection_counts=rejected,
        summary=dict(formals=len(formals), matched_formals=len(matched), observed_operand_links=len(links)),
        scope='FORMAL_ENTRY_REFERENCE_TO_SAME_CAPTURE_NATIVE_OPERAND_OBSERVATION',
        caller_actual_path_established=False, logical_content_versions_established=False,
        object_storage_continuity_established=False, effects_closed=False, transformation_authorized=False)

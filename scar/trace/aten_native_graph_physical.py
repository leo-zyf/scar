"""Recover exact input geometry for a replayed scoped native ATen graph."""

from scar.ir.record_codec import encode
from scar.trace.capture_coordinates import CaptureCoordinates, bind_capture_coordinates
from scar.trace.native_tensor_inputs import NativeInputPayloadBudget, prepare_native_input_metadata


def _require(condition, reason):
    if not condition:
        raise ValueError('ATen graph input metadata: ' + reason)


def _matched_rows(coordinates):
    capability = coordinates.report.get('native_tensor_input_membership') or {}
    if capability.get('capability') != 'SCAR_NATIVE_TENSOR_INPUT_MEMBERSHIP_V1':
        return {}, 'NATIVE_INPUT_COMPANION_UNAVAILABLE'

    metadata = prepare_native_input_metadata(coordinates, budget=NativeInputPayloadBudget()).report()
    rows = {}
    for row in metadata['rows']:
        if 'original_node' not in row:
            continue
        locator = row['original_node']['locator']
        _require(locator not in rows, 'ambiguous original operator')
        rows[locator] = row
    return rows, 'EXACT_TORCHOP_INPUT_JOIN_UNAVAILABLE'


def _ports(op, key):
    for port in op[key]:
        if port['kind'] != 'TENSOR_REFERENCE':
            continue
        locator = f"{op['eto']['locator']}/{key}/values/{port['ordinal']}"
        _require(port['source'] == dict(artifact=op['eto']['artifact'], locator=locator),
                 'scoped root does not match its original operator path')
        yield port


def recover_graph_input_metadata(capture, native):
    """Join already replayed ETO roots to same-run TorchOp boxed input metadata.

    The native input reader has already checked the event correlation, original
    RecordFunction ID, and geometry of each exact Tensor path. This view further
    requires the scoped operator and Kineto fact references to be identical.
    Physical metadata describes that input event alone. An observed output
    descriptor carries its original ETO type and geometry, with no layout claim.
    """
    coordinates = capture if type(capture) is CaptureCoordinates else bind_capture_coordinates(capture)
    _require(native['schema'] == 'scar.aten-scoped-ports' and native['schema_version'] == 1
             and native['run'] == encode(coordinates.run.id)
             and native['process'] == encode(coordinates.target_process.id)
             and native['coordinate_binding'] == encode(coordinates.artifact),
             'scoped ports do not belong to this capture')
    rows, missing_reason = _matched_rows(coordinates)
    inputs, outputs, unresolved = [], [], []
    for op in native['operators']:
        node = rows.get(op['eto']['locator'])
        if node is not None:
            _require(node['original_node'] == op['eto'] and node['CPU_bridge'] == op['kineto'],
                     'scoped operator and metadata join refer to different original events')
        leaves = {} if node is None else {tuple(row['path']): row for row in node['metadata']}
        for port in _ports(op, 'inputs'):
            leaf = leaves.get((port['ordinal'],))
            if leaf is None:
                unresolved.append(dict(port=port['source'], reason=missing_reason if node is None
                                       else 'TENSOR_ROOT_HAS_NO_EXPOSED_METADATA'))
                continue
            anchor = dict(metadata=leaf['source'], kineto=node['CPU_bridge'], eto_node=node['original_node'])
            info = leaf['metadata']
            fields = (info['sizes'], info['strides'], info['device'], info['dtype'], info['layout'])
            if any(value is None for value in fields):
                unresolved.append(dict(port=port['source'], reason='INCOMPLETE_TENSOR_METADATA', evidence=anchor))
                continue
            device = info['device']['text']
            _require((info['sizes'], info['strides'], device) == (
                     port['shape'], port['strides'], port['tensor_descriptor'][5]),
                     'physical metadata shape, strides or device differs from the scoped root')
            inputs.append(dict(port=port['source'], descriptor_id=port['descriptor_id'],
                dtype=info['dtype'].removeprefix('torch.'), device=device,
                layout=info['layout'].removeprefix('torch.'), shape=info['sizes'], strides=info['strides'],
                evidence=anchor))
        for port in _ports(op, 'outputs'):
            outputs.append(dict(port=port['source'], descriptor_id=port['descriptor_id'],
                provider_type=port['provider_type'], device=port['tensor_descriptor'][5],
                shape=port['shape'], strides=port['strides'], layout=None))
    return dict(schema='scar.aten-native-graph-physical', schema_version=1,
        run=native['run'], process=native['process'], coordinate_binding=native['coordinate_binding'],
        inputs=inputs, outputs=outputs, unresolved=unresolved,
        native_outputs='ETO_DESCRIPTORS_ONLY', logical_content_established=False,
        resource_readiness_established=False, source_replacement_authorized=False)

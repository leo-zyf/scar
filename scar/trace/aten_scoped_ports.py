"""Recover ordered ATen root port observations inside original CALL scopes."""

from dataclasses import dataclass, replace
from itertools import islice

from scar.ir.record_codec import encode
from scar.trace.capture_coordinates import bind_capture_coordinates
from scar.trace.native_call_scope_import import _member, import_native_call_scopes
from scar.trace.native_import import _attrs, _load_importer, _read
from scar.trace.native_scalar_content import SCALAR_CONTENT_PROFILE_V2, ScalarFrontier, decode_pinned_scalar
from scar.trace.native_schema import _unwrap_optional, parse_schema_ports
from scar.trace.torch_call_scopes import LIMIT


_MAX_BYTES = 128_000_000
_MAX_LINES = 2_000_000
_MAX_SLOTS = 1024
_MALFORMED_SCALAR = frozenset({ScalarFrontier.FORMAL_KIND, ScalarFrontier.PAYLOAD_TYPE,
                             ScalarFrontier.INT64_RANGE, ScalarFrontier.NONE_SENTINEL})
_CONTAINER_KINDS = (('Tuple[', 'TupleType'), ('GenericTuple[', 'TupleType'), ('List[', 'ListType'),
                    ('GenericList', 'ListType'), ('Dict[', 'DictType'), ('GenericDict', 'DictType'))


def _capture_member(coordinates, schema):
    matches = [row['logical_path'] for row in coordinates.report['membership']
               if row['artifact']['schema'] == schema]
    if len(matches) != 1:
        raise ValueError('expected one original capture member for ' + schema)
    return _member(coordinates, matches[0], schema, coordinates.target_process.pid)


def _tensor_port(slot, value, shape, strides, original_slot, kind, real_kind):
    if kind != 'TensorType' or real_kind != 'TensorType':
        raise ValueError('ETO Tensor port is outside its schema Tensor formal')
    if (type(value) is not list or len(value) != 6
            or any(type(n) is not int or not 0 <= n < 2**64 for n in value[:5])
            or type(value[5]) is not str or not 0 < len(value[5]) <= 128):
        raise ValueError('malformed original ETO Tensor descriptor')
    if (type(shape) is not list or type(strides) is not list or len(shape) != len(strides) or len(shape) > 64
            or any(type(n) is not int or not 0 <= n < 2**63 for n in (*shape, *strides))):
        raise ValueError('malformed original ETO Tensor geometry')
    locator = slot['source']['locator']
    artifact = slot['source']['artifact']
    if original_slot is None or original_slot['raw_reference'] != 'sha256:' + artifact['sha256'] + locator:
        raise ValueError('ETO Tensor root has no consistent native import reference')
    return dict(slot, kind='TENSOR_REFERENCE', tensor_descriptor=value.copy(), shape=shape.copy(),
                strides=strides.copy(), provider_reference=original_slot['provider_reference'],
                descriptor_id=original_slot['descriptor_id'])


def _root_ports(node, role, formals, observed_slots, source, revision):
    record = node.get(role)
    columns = ('values', 'types', 'shapes', 'strides')
    if (len(formals) > _MAX_SLOTS or type(record) is not dict
            or any(type(record.get(key)) is not list or len(record[key]) != len(formals) for key in columns)):
        raise ValueError('original ETO root port inventory differs from schema arity')
    result = []
    for index, (formal, value, tag, shape, strides) in enumerate(zip(formals, *(record[key] for key in columns))):
        if type(tag) is not str or not 0 < len(tag) <= 4096:
            raise ValueError('original ETO root port requires a bounded provider tag')
        slot = dict(ordinal=index, formal_name=formal.name, provider_type=tag,
                    source=dict(artifact=source['artifact'], locator=f"{source['locator']}/{role}/values/{index}"))
        original_slot = observed_slots.get((role, (index,)))
        absent = tag == 'None'
        kind = (formal.type if absent else _unwrap_optional(formal.type)).kind()
        real_kind = (formal.real_type if absent else _unwrap_optional(formal.real_type)).kind()
        if tag == 'Tensor' or tag.startswith('Tensor('):
            if tag != 'Tensor' and (len(tag) <= len('Tensor()') or not tag.endswith(')')):
                raise ValueError('malformed original ETO Tensor tag')
            result.append(_tensor_port(slot, value, shape, strides, original_slot, kind, real_kind))
            continue
        if original_slot is not None or kind == 'TensorType' or real_kind == 'TensorType':
            raise ValueError('non-Tensor ETO port conflicts with its Tensor formal or reference')
        for prefix, container_kind in _CONTAINER_KINDS:
            if tag.startswith(prefix) and (kind != container_kind or real_kind != container_kind):
                raise ValueError('ETO container root conflicts with its schema formal branch')
        scalar = decode_pinned_scalar(revision, tag, kind, value,
                                     formal_real_kind=real_kind, profile=SCALAR_CONTENT_PROFILE_V2)
        if scalar.frontier in _MALFORMED_SCALAR:
            raise ValueError('malformed original ETO scalar port: ' + scalar.frontier.value)
        if scalar.literal is None:
            result.append(dict(slot, kind='UNRESOLVED', frontier=scalar.frontier.value))
        else:
            result.append(dict(slot, kind='EXACT_SCALAR', literal=encode(scalar.literal)))
    return result


@dataclass(frozen=True, slots=True)
class ScopedAtenCapture:
    coordinates: object
    importer: object
    scopes: dict
    calls: tuple[dict, ...]
    revision: str


def _attempts(call_attempts):
    attempts = tuple(islice(call_attempts, LIMIT + 1))
    if (not attempts or len(attempts) > LIMIT or any(type(n) is not int or n < 1 for n in attempts)
            or len(set(attempts)) != len(attempts)):
        raise ValueError('select bounded unique positive original CALL attempts')
    return attempts


def _select_calls(capture, attempts):
    scoped_calls = {row['attempt']: row for row in capture.calls}
    calls = []
    for attempt in attempts:
        call = scoped_calls.get(attempt)
        if call is None:
            raise ValueError(f'CALL attempt {attempt} has no normally closed, joined operator scope')
        calls.append(call)
    return replace(capture, calls=tuple(calls))


def read_scoped_aten_capture(coordinates, *, call_attempts=None, max_bytes=_MAX_BYTES, max_lines=_MAX_LINES):
    """Replay original joined scopes once before loading the provider schema parser.

    Without an explicit selection, return eligible observations in original
    attempt order. The scope report retains the collector gaps and budget state.
    """
    attempts = None if call_attempts is None else _attempts(call_attempts)
    if any(type(n) is not int or n < 1 for n in (max_bytes, max_lines)):
        raise ValueError('scoped port byte and line budgets must be positive exact integers')

    manifest, manifest_path = _capture_member(coordinates, 'scar.execution-evidence-collection')
    _, eto_path = _capture_member(coordinates, 'pytorch.execution-trace.v1')
    _, kineto_path = _capture_member(coordinates, 'pytorch.kineto.chrome-trace.v1')

    declaration, identity = _read(manifest_path, min(max_bytes, 2_000_000))
    if (identity['sha256'], identity['size_bytes']) != (manifest.sha256, manifest.size_bytes):
        raise ValueError('collector manifest changed after capture binding')
    revision = declaration['provider']['torch_git_version']
    importer = _load_importer(eto_path, kineto_path, max_bytes=max_bytes, provider_revision=revision)
    scopes = import_native_call_scopes(importer.finish(), manifest_path, coordinates,
                                       max_bytes=max_bytes, max_lines=max_lines)
    calls = tuple(sorted((row for row in scopes['observations']
                          if row['outcome'] == 'C_RETURN' and row['status'] == 'CLOSED_SCOPE_OBSERVED'
                          and row['operators']), key=lambda row: row['attempt']))
    capture = ScopedAtenCapture(coordinates, importer, scopes, calls, revision)
    return capture if attempts is None else _select_calls(capture, attempts)


def decode_scoped_aten_ports(capture):
    """Decode the checked scope records after the source-only F4 reader has closed."""
    coordinates, importer, scopes, revision = capture.coordinates, capture.importer, capture.scopes, capture.revision
    schemas, operators = {}, []
    for call in capture.calls:
        for op in call['operators']:
            raw_schema, source = op['schema'], op['eto']
            ordinal = int(source['locator'].removeprefix('#/nodes/'))
            node = importer.eto['nodes'][ordinal]
            if (raw_schema != _attrs(node).get('op_schema') or node['id'] != op['provider_node_id']
                    or not raw_schema.startswith('aten::') or '...' in raw_schema):
                raise ValueError('original scoped operator has no supported ATen schema')
            if raw_schema not in schemas:
                schemas[raw_schema] = parse_schema_ports(raw_schema, provider_revision=revision)
            schema, arguments, returns = schemas[raw_schema]
            if schema.name != node['name']:
                raise ValueError('ATen schema name changes its original ETO operator')
            observed_slots = {
                (slot['role'], tuple(slot['ordered_path'])): slot
                for slot in importer.nodes[node['id']].metadata.get('tensor_slots', ())}
            inputs = _root_ports(node, 'inputs', schema.arguments, observed_slots, source, revision)
            outputs = _root_ports(node, 'outputs', schema.returns, observed_slots, source, revision)
            declared_aliases = [
                dict(direction=arg['direction'], ordinal=arg['ordinal'], alias=arg['alias'],
                     port=port['source'], is_out=arg['is_out'])
                for args, ports in ((arguments, inputs), (returns, outputs))
                for arg, port in zip(args, ports, strict=True) if arg['alias'] is not None
            ]
            operators.append(dict(call_attempt=call['attempt'], call_site=call['site'], invocation=call['invocation'],
                operand=call['operand'], scope=call['declaration'], eto=source, kineto=op['kineto'],
                schema=raw_schema, inputs=inputs, outputs=outputs,
                declared_schema_effects=dict(may_mutate=bool(schema.is_mutable), alias_ports=declared_aliases)))

    return dict(schema='scar.aten-scoped-ports', schema_version=1, run=scopes['run'], process=scopes['process'],
        provider_revision=revision, coordinate_binding=encode(coordinates.artifact), operators=operators,
        reference_scope='SAME_CAPTURE_PROVIDER_SYMBOLS', all_children_covered=False,
        logical_values_established=False, source_argument_mapping_established=False,
        resource_readiness_established=False, effects_established=False, transformation_authorized=False)


def recover_scoped_aten_ports(coordinates, *, call_attempts, max_bytes=_MAX_BYTES, max_lines=_MAX_LINES):
    """Replay CALL containment and recover ordered Tensor reference symbols and boxed scalars."""
    scope = read_scoped_aten_capture(coordinates, call_attempts=call_attempts,
                                     max_bytes=max_bytes, max_lines=max_lines)
    return decode_scoped_aten_ports(scope)


def read_scoped_aten_ports(capture_root, *, call_attempts, max_bytes=_MAX_BYTES, max_lines=_MAX_LINES):
    coordinates = bind_capture_coordinates(capture_root, max_bytes=max_bytes, max_lines=max_lines)
    return recover_scoped_aten_ports(coordinates, call_attempts=call_attempts,
                                     max_bytes=max_bytes, max_lines=max_lines)

"""Freeze a bit-exact compact Basis port for a private native handoff."""

import argparse
import json
from pathlib import Path
from struct import pack

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import native_region_handoff as handoff

SCHEMA = 'scar.native-compact-basis-proof'
WIRE = 'scar.native-compact-basis-wire'
MAGIC = b'SCARBAS2'
_DENSE_ROLES = ('grad', 'value', 'quadrature', 'dofs', 'constraints')


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _derive(handoff_path, producer_path):
    from scar.codegen.native_mesh_basis import MESH_SCHEMA, read_package

    source_ref = _ref(handoff_path, handoff.SCHEMA)
    source = handoff.check(source_ref['uri'])
    capture_ref = source['inputs']['capture']
    capture = read_artifact(capture_ref)
    witness = read_artifact(source['inputs']['basis_witness'])
    producer = _ref(Path(producer_path) / 'artifact.json', MESH_SCHEMA)
    consumer = source['inputs']['basis_package']
    _, native_producer = read_package(Path(producer['uri']).parent, 'mesh')
    _, native_consumer = read_package(Path(consumer['uri']).parent, 'basis')
    dense_ref = native_consumer['admitted_input']
    _require(native_producer['schema_version'] == native_consumer['schema_version'] == 1
        and native_producer['capture'] == capture_ref
        and native_producer['basis_package'] == consumer and native_producer['exact_basis'] == dense_ref
        and native_producer['admitted_input'] == capture['mesh_wire']
        and capture['expected_basis'] == dense_ref and native_consumer['witness'] == source['inputs']['basis_witness']
        and native_producer['libraries'] == native_consumer['libraries'],
        'compact layout requires the same native producer, consumer and numeric ABI')
    dense = read_artifact(dense_ref, json_document=False)
    mesh_bytes = read_artifact(capture['mesh_wire'], json_document=False)
    header = bytes(native_consumer['plan']['input_header'])
    offset = handoff.HEADER.size + len(header)
    _require(len(header) == 100 and dense.startswith(header) and header[:8] == b'SCARBAS1'
        and mesh_bytes[handoff.HEADER.size:offset] == header,
        'legacy source and consumer must use the same checked wire header')
    original = handoff._payload(dense, len(header), ((role, witness['fields'][role]) for role in _DENSE_ROLES))
    mesh_fields = handoff._payload(mesh_bytes, offset, ((role,capture['fields'][role]) for role in handoff.MESH_FIELDS))
    view = handoff._representations(capture, mesh_fields, witness, original)
    _require(view == source['broadcast_view'] and view['broadcast_axis'] == 1
        and view['borrowed_strides_bytes'] == [native_consumer['plan']['points']*8, 0, 8]
        and view['dtype'] == '<f8',
        'compact field cannot reconstruct the qualified dense view without changing bytes')
    strides = pack('<3I', *view['borrowed_strides_bytes'])
    compact = b''.join((MAGIC, header[8:], strides, original['grad'], mesh_fields['reference_values'],
                        original['quadrature'], original['dofs'], original['constraints']))
    _require(len(dense) - len(compact) == view['serialized_payload_bytes_avoidable_with_compact_port'] - len(strides),
             'compact byte count differs from the proved representation and its metadata')
    placement = source['native_placement']
    proof = dict(schema=SCHEMA, schema_version=1, handoff=source_ref, producer=producer, consumer=consumer,
        mesh_capture=capture_ref, dense_input=dense_ref,
        wire=dict(magic=MAGIC.decode(), value_strides_bytes=view['borrowed_strides_bytes'],
            dense_bytes=len(dense), compact_bytes=len(compact), basis_header_bytes=len(header),
            stride_header_bytes=len(strides), serialized_bytes_saved=len(dense)-len(compact),
            role_order=list(_DENSE_ROLES), logical_value_shape=view['expanded_shape'],
            stored_value_shape=capture['fields']['reference_values']['shape']),
        placement=dict(owner_cpu=placement['owner_cpu'], numa_node=placement['numa_node'], execution_capacity=1,
            storage='PRIVATE_HOST_OWNER_ALLOCATION', physical_page_residency_proven=False),
        numeric_contract=dict(dtype=view['dtype'], exact_logical_value_bytes=True,
            unchanged_roles=['grad', 'quadrature', 'dofs', 'constraints'],
            point_order=native_consumer['plan']['point_order'],
            component_order=native_consumer['plan']['component_order'],
            value_scale=native_consumer['plan']['value_scale'],
            required_rounding='SAME_NATIVE_PROVIDER_AND_DEFAULT_BINARY64_NO_FP_CONTRACTION'),
        legality=dict(status='QUALIFIED_WIRE_BYTES_FOR_BACKEND_VALIDATION',
            scope='PRIVATE_NATIVE_FIXED_INPUT_PORT',
            consumer_guards=['OWN_COMPACT_BACKING_THROUGH_LAST_USE', 'READ_ONLY_ZERO_STRIDE_VIEW',
                             'SAME_FAN_OUT_AND_OUTPUTS', 'UNCHANGED_REDUCTION_AND_FP_CONTROL',
                             'NO_EXTERNAL_REFERENCE_TO_PRIVATE_REPRESENTATION'],
            preserves_original_mesh_and_basis_consumers=True,
            mutation_escape_proof='REQUIRED_FOR_COMPACT_NATIVE_CONSUMER',
            original_python_effects_discharged=False, future_input_invariance_proven=False),
        decision=dict(selected='LEGACY_DENSE', matched_native_time_ns=None,
            reason='CONTROLLED_MATCHED_NATIVE_MEASUREMENT_REQUIRED', deployment_authorized=False))
    return proof, compact


def _bind(proof, expected, path):
    reference = _ref(path, WIRE)
    _require(read_artifact(reference, json_document=False) == expected,
             'compact Basis wire differs from its exact reconstruction')
    return dict(proof, compact_input=reference)


def check(path):
    frozen = read_artifact(_ref(path, SCHEMA))
    proof, expected = _derive(frozen['handoff']['uri'], Path(frozen['producer']['uri']).parent)
    _require(frozen == _bind(proof, expected, frozen['compact_input']['uri']),
             'compact Basis proof differs from its cold replay')
    return frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', type=Path)
    for name in ('handoff', 'producer', 'compact', 'out'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args(argv)
    if args.check:
        print(check(args.check)['wire']['compact_bytes'])
        return
    if not all((args.handoff, args.producer, args.compact, args.out)):
        parser.error('provide handoff, legacy producer package, compact output and proof output')
    _require(not args.compact.exists() and not args.out.exists(), 'fresh compact outputs required')
    proof, expected = _derive(args.handoff, args.producer)
    with args.compact.open('xb') as stream:
        stream.write(expected)
    proof = _bind(proof, expected, args.compact)
    with args.out.open('x') as stream:
        json.dump(proof, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()

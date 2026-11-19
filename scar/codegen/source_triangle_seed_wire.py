"""Encode source-derived scalar constants using the admitted native topology contract."""

from hashlib import sha256
from math import isfinite, prod
from struct import Struct, pack

SOURCE = 'scar.source-triangle-seed'
_SEED = Struct('<8s4I')
_MESH = Struct('<8s3I')
_BASIS = Struct('<8s4I')
_COO = Struct('<8s3I')
_CSR = Struct('<8s32s4I')
_BASIS_OFFSET = _SEED.size + _MESH.size
_CSR_OFFSET = _BASIS_OFFSET + _BASIS.size + _COO.size
_MAX_INLINE_BYTES = 4096


def _require(valid, reason):
    if not valid:
        raise ValueError(reason)


def wire_payload(derived, topology, release):
    """Use only the retained header, source constants and sealed content identity."""
    _require(derived['schema'] == SOURCE and derived['schema_version'] == 1
        and derived['entry'] == release['entry'] and derived['argv'] == release['context']['argv']
        and type(derived['iterations']) is int and derived['iterations'] > 0
        and derived['argv'] == [str(derived['iterations'])], 'source seed belongs to another fixed entry contract')
    header = bytes(topology['plan']['header'])
    tag, vertices, cells, depth, flags = _SEED.unpack_from(header)
    basis, functions, components, _, points = _BASIS.unpack_from(header, _BASIS_OFFSET)
    csr, _, _, _, rounds, _ = _CSR.unpack_from(header, _CSR_OFFSET)
    _require(len(header) == _CSR_OFFSET + _CSR.size and tag == b'SCARTRI1' and flags == 0
        and basis == b'SCARBAS1' and csr == b'SCARPSR1' and (functions, components) == (3, 2)
        and derived['depth'] == depth and derived['iterations'] == rounds == release['count'],
        'source refinement, element or count does not match the native launch ABI')
    specs = (
        ('seed_coordinates', '<f8', [2, vertices]),
        ('seed_connectivity', '<i4', [3, cells]),
        ('reference_values', '<f8', [functions, points]),
        ('reference_derivatives', '<f8', [functions, components, points]),
        ('quadrature_weights', '<f8', [points]),
    )
    fields = derived['fields']
    _require(set(fields) == {name for name, _, _ in specs}, 'source seed requires the exact primitive operand roles')
    payload = bytearray(header)
    for name, dtype, shape in specs:
        field = fields[name]
        values = field['values']
        _require(field['shape'] == shape and field['dtype'] == dtype and len(values) == prod(shape),
            'source operand shape or type differs: ' + name)
        if dtype == '<i4':
            _require(all(type(value) is int and -2**31 <= value < 2**31 for value in values),
                'expected exact signed integer sources')
            payload.extend(pack(f'<{len(values)}i', *values))
        else:
            _require(all(type(value) is str for value in values), 'expected canonical binary64 hex sources')
            floats = tuple(map(float.fromhex, values))
            _require(all(isfinite(value) and value.hex() == word for value, word in zip(floats, values, strict=True)),
                'expected finite canonical binary64 sources')
            payload.extend(pack(f'<{len(floats)}d', *floats))
    original = topology['admitted_input']
    _require(0 < len(payload) <= _MAX_INLINE_BYTES and original['schema'] == 'scar.native-triangular-seed-wire'
        and original['size_bytes'] == len(payload) and original['sha256'] == sha256(payload).hexdigest(),
        'source-generated seed differs from the selected native ingress identity')
    return bytes(payload)

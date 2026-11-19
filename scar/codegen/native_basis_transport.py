"""Bind the compact read-only Basis wire to a verified Stage II publication proof."""

from pathlib import Path
from struct import Struct

from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.native-compact-basis-proof'
MEMBER = 'compact-proof.json'
_SHAPE = Struct('<8s4I')
_STRIDES = Struct('<3I')


def read_compact(path):
    from scar.optimization.native_compact_basis import check

    reference = artifact_reference(Path(path).resolve(strict=True), SCHEMA)
    value = check(reference['uri'])
    read_artifact(reference, json_document=False)
    return reference, value


def project_header(dense, wire):
    magic, basis, _, elements, points = _SHAPE.unpack_from(dense)
    strides = (points * 8, 0, 8)
    if (magic != b'SCARBAS1' or len(dense) != wire['basis_header_bytes'] or len(dense) != 100
            or wire['magic'] != 'SCARBAS2' or wire['value_strides_bytes'] != list(strides)
            or wire['stride_header_bytes'] != _STRIDES.size
            or wire['dense_bytes'] - wire['compact_bytes'] != basis * (elements - 1) * points * 8 - _STRIDES.size):
        raise ValueError('compact Basis descriptor does not fit the exact dense header')
    return b'SCARBAS2' + dense[8:] + _STRIDES.pack(*strides)


def write_proof(root, ref):
    member = root / MEMBER
    member.write_bytes(read_artifact(ref, json_document=False))
    return {**artifact_reference(member, SCHEMA), 'uri': MEMBER}


def read_local_proof(root, doc):
    member = root / MEMBER
    ref = {**artifact_reference(member, SCHEMA), 'uri': MEMBER}
    if (member.resolve(strict=True) != member or doc['proof_member'] != ref
            or {**doc['compact_proof'], 'uri': MEMBER} != ref):
        raise ValueError('packed Basis proof differs from its original qualified identity')
    result = read_artifact({**ref, 'uri': str(member)})
    if result['wire'] != doc['wire']:
        raise ValueError('packed Basis proof wire differs from the compiled ABI')
    return result

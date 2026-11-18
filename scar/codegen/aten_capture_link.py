"""Resolve exact retained ETO occurrences to the installed ATen C++ interface."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from scar.codegen.aten_provider import link_aten_provider
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import decode, encode, loads
from scar.trace.capture_coordinates import bind_capture_coordinates
from scar.trace.native_import import _attrs, _load_importer, _read, import_native_evidence


_MODEL = 'scar.capture-model'
_NODE = re.compile(r'#/nodes/(0|[1-9][0-9]*)')
_MAX_MODEL_BYTES = 4_000_000
_MAX_ETO_BYTES = 64_000_000


def _member(coordinates, schema):
    matches = [row for row in coordinates.report['membership'] if row['artifact']['schema'] == schema]
    if len(matches) != 1:
        raise ValueError('one original capture member required: ' + schema)
    row, = matches
    artifact = decode(ProviderArtifact, row['artifact'])
    if coordinates.process_for(artifact, coordinates.target_process.pid) != coordinates.target_process:
        raise ValueError('capture member cannot be bound to the target process: ' + schema)
    return artifact, row['local_path']


def _check_member(artifact, actual):
    if (artifact.size_bytes, artifact.sha256) != (actual['size_bytes'], actual['sha256']):
        raise ValueError('original capture member differs: ' + artifact.uri)


def _scoped_operators(coordinates, manifest_path, et_path, revision, attempts):
    from scar.trace.native_call_scope_import import import_native_call_scopes

    kineto, kineto_path = _member(coordinates, 'pytorch.kineto.chrome-trace.v1')
    if kineto.size_bytes > _MAX_ETO_BYTES:
        raise ValueError('decoded Kineto exceeds the retained linkage bound')
    native = import_native_evidence(et_path, kineto_path, max_bytes=2 * _MAX_ETO_BYTES,
                                    provider_revision=revision)
    scopes = import_native_call_scopes(native, manifest_path, coordinates,
                                      max_bytes=2 * _MAX_ETO_BYTES, max_lines=2_000_000)
    rows = {row['attempt']: row for row in scopes['observations']}
    selected = []
    for attempt in attempts:
        row = rows.get(attempt)
        if (row is None or row['status'] != 'CLOSED_SCOPE_OBSERVED'
                or row['outcome'] != 'C_RETURN' or not row['operators']):
            raise ValueError(f'original CALL attempt {attempt} has no normally closed, joined operator scope')
        for op in row['operators']:
            selected.append(dict(source=op['eto'], provider_node_id=op['provider_node_id'], schema=op['schema'],
                call_attempt=attempt, operand=row['operand'], call_site=row['site'],
                scope=row['declaration'], kineto=op['kineto']))
    return selected

def _located_operators(coordinates, et_artifact, et_path, revision, locators):
    # Run the existing original ETO parser without constructing a new graph.
    importer = _load_importer(et_path, None, max_bytes=et_artifact.size_bytes, provider_revision=revision)
    _check_member(et_artifact, importer.artifacts['eto'])
    if importer.pid != coordinates.target_process.pid:
        raise ValueError('ETO PID differs from the bound target process')
    nodes = importer.eto['nodes']
    ids = set()
    for node in nodes:
        if type(node) is not dict or type(node.get('id')) is not int or node['id'] < 0 or node['id'] in ids:
            raise ValueError('malformed or duplicate original ETO node ID')
        ids.add(node['id'])

    selected = []
    for locator in locators:
        ordinal = int(_NODE.fullmatch(locator)[1])
        if ordinal >= len(nodes):
            raise ValueError('ETO node ordinal is outside the original capture: ' + locator)
        row = nodes[ordinal]
        schema = _attrs(row).get('op_schema')
        if (type(schema) is not str or not schema.startswith('aten::')
                or schema.split('(', 1)[0].split('.', 1)[0] != row.get('name')):
            raise ValueError('original ETO node has no matching ATen schema: ' + locator)
        selected.append(dict(source=encode(ProviderFactReference(et_artifact, locator)),
                             provider_node_id=row['id'], schema=schema))
    return selected


def select_aten_capture(capture_model, *, expected_sha256, node_locators=(), call_attempts=()):
    """Read original ATen declarations from ETO pointers or observed CALL attempts."""
    if type(expected_sha256) is not str or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('capture model requires its exact SHA256')
    locators, attempts = tuple(node_locators), tuple(call_attempts)
    if bool(locators) == bool(attempts):
        raise ValueError('select exact ETO pointers or CALL attempts')
    if locators and (any(type(item) is not str or not _NODE.fullmatch(item) for item in locators)
                     or len(set(locators)) != len(locators)):
        raise ValueError('select unique original #/nodes/N ETO pointers')
    if attempts and (any(type(item) is not int or item < 1 for item in attempts)
                      or len(set(attempts)) != len(attempts)):
        raise ValueError('select unique positive original CALL attempts')

    path = Path(capture_model).resolve()
    with path.open('rb') as stream:
        raw = stream.read(_MAX_MODEL_BYTES + 1)
    if len(raw) > _MAX_MODEL_BYTES or sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('capture model bytes differ or exceed the metadata bound')
    model = loads(raw)
    if (type(model) is not dict or model.get('schema') != _MODEL or model.get('schema_version') != 1
            or model.get('coordinate_identity_consistent') is not True):
        raise ValueError('unsupported retained capture model')
    broker = decode(ProviderArtifact, model['coordinate_binding'])
    coordinates = bind_capture_coordinates(model['capture'], profile_uri=broker.uri)
    if coordinates.artifact != broker or encode(coordinates.run) != model['run']:
        raise ValueError('capture model differs from freshly verified capture coordinates')

    manifest, manifest_path = _member(coordinates, 'scar.execution-evidence-collection')
    provider, actual = _read(manifest_path, manifest.size_bytes)
    _check_member(manifest, actual)
    et_artifact, et_path = _member(coordinates, 'pytorch.execution-trace.v1')
    revision = provider['provider']['torch_git_version']
    if provider['artifacts']['execution_trace']['path'] != et_artifact.uri:
        raise ValueError('native collector declares a different ETO source')
    if et_artifact.size_bytes > _MAX_ETO_BYTES:
        raise ValueError('decoded ETO exceeds the retained linkage bound')
    selected = (_scoped_operators(coordinates, manifest_path, et_path, revision, attempts)
                if attempts else _located_operators(coordinates, et_artifact, et_path, revision, locators))
    return dict(schema='scar.aten-capture-source', schema_version=1,
        capture_model=encode(ProviderArtifact(str(path), expected_sha256, len(raw), _MODEL)),
        coordinate_binding=encode(broker), collector_manifest=encode(manifest),
        run=encode(coordinates.run.id), process=encode(coordinates.target_process.id),
        provider_revision=revision, occurrences=selected,
        dispatch_device_proven=False, replacement_authorized=False)


def link_aten_capture(capture_model, out, *, expected_sha256, node_locators=(), call_attempts=()):
    """Publish existing typed provider linkage with original occurrence receipts."""
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    source = select_aten_capture(capture_model, expected_sha256=expected_sha256,
                                 node_locators=node_locators, call_attempts=call_attempts)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-aten-capture-', dir=out.parent) as temporary:
        linked = Path(temporary) / 'linked'
        result = link_aten_provider([row['schema'] for row in source['occurrences']], linked,
                                    provider_revision=source['provider_revision'])
        entries = {row['schema']: row['entry'] for row in result['operators']}
        for row in source['occurrences']:
            row['entry'] = entries[row['schema']]
        report_bytes = (linked / 'result.json').read_bytes()
        source['linkage_report'] = encode(ProviderArtifact(str(out / 'result.json'),
            sha256(report_bytes).hexdigest(), len(report_bytes), result['schema']))
        (linked / 'capture-source.json').write_text(json.dumps(source, sort_keys=True, indent=2) + '\n')
        linked.rename(out)
    return result, source


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar link-aten-capture', description=__doc__)
    parser.add_argument('--capture-model', required=True)
    parser.add_argument('--capture-model-sha256', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--eto-node', action='append', default=[])
    source.add_argument('--call-attempt', action='append', type=int, default=[])
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    report, source = link_aten_capture(args.capture_model, args.out,
        expected_sha256=args.capture_model_sha256, node_locators=args.eto_node, call_attempts=args.call_attempt)
    print(json.dumps(dict(status=report['status'], operators=len(report['operators']),
        occurrences=len(source['occurrences']), source=str(Path(args.out) / 'capture-source.json'))))
    return 0

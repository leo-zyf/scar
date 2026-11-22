"""Versioned macro execution plans shared by costing and backend lowering.

The envelope references source/analysis artifacts and an existing typed task
graph. It carries no timings or selection result: those are separate records
keyed by recipe_id. Profile adapters own semantic and backend applicability.
"""
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re

SCHEMA = 'scar.execution-recipe'
VERSION = 1
_LIMIT = 64 * 1024 * 1024
_FIELDS = {'schema', 'schema_version', 'recipe_id', 'profile', 'semantic_boundary',
           'control', 'task_graph', 'resources', 'implementation', 'artifacts', 'obligations'}
_ARTIFACT_FIELDS = {'uri', 'sha256', 'size_bytes', 'schema'}
_OBLIGATION_STATES = {'PROVEN', 'RUNTIME_GUARD', 'EXTERNAL_ASSUMPTION', 'UNRESOLVED', 'LOWERING_CHECK'}


def _copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _pairs(rows):
    result = {}
    for name, value in rows:
        if name in result: raise ValueError('duplicate recipe JSON field: ' + name)
        result[name] = value
    return result


def _invalid(value):
    raise ValueError('nonfinite recipe JSON value: ' + value)


def _json(raw):
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_invalid)


def _bounded_bytes(stream):
    raw = stream.read(_LIMIT + 1)
    if len(raw) > _LIMIT: raise ValueError('recipe component exceeds 64 MiB; reference a smaller shard')
    return raw


def _bytes(path):
    with Path(path).open('rb') as stream:
        return _bounded_bytes(stream)


def _reference(value):
    value = asdict(value) if is_dataclass(value) else deepcopy(value)
    if (type(value) is not dict or set(value) != _ARTIFACT_FIELDS
            or type(value['uri']) is not str or not Path(value['uri']).is_absolute()
            or type(value['sha256']) is not str or re.fullmatch('[0-9a-f]{64}', value['sha256']) is None
            or type(value['size_bytes']) is not int or value['size_bytes'] < 0
            or type(value['schema']) is not str or not value['schema']):
        raise ValueError('complete original ProviderArtifact reference required')
    return value


def _reference_from_bytes(path, schema, raw):
    return _reference(dict(uri=str(path), sha256=sha256(raw).hexdigest(),
                           size_bytes=len(raw), schema=schema))


def artifact_reference(path, schema):
    path = Path(path).absolute()
    return _reference_from_bytes(path, schema, _bytes(path))


def _file_identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _artifact_document(raw, path, schema):
    document = _json(raw)
    actual = document.get('schema') if type(document) is dict else None
    if type(actual) is not str or not actual or (schema is not None and actual != schema):
        raise ValueError('recipe component schema differs: ' + str(path))
    return document


def open_artifact(path, schema=None):
    """Open a previously unsealed document and bind its receipt to the parsed bytes."""
    path = Path(path).absolute()
    with path.open('rb') as stream:
        original = _file_identity(os.fstat(stream.fileno()))
        raw = _bounded_bytes(stream)
        if (original != _file_identity(os.fstat(stream.fileno()))
                or original != _file_identity(path.stat())):
            raise ValueError('recipe component changed while opening: ' + str(path))
    document = _artifact_document(raw, path, schema)
    return _reference_from_bytes(path, document['schema'], raw), document


def read_artifact(reference, *, json_document=True):
    reference = _reference(reference); raw = _bytes(reference['uri'])
    if len(raw) != reference['size_bytes'] or sha256(raw).hexdigest() != reference['sha256']:
        raise ValueError('recipe component byte identity differs: ' + reference['uri'])
    if not json_document: return raw
    return _artifact_document(raw, reference['uri'], reference['schema'])


def recipe_id(recipe):
    payload = _copy(recipe); payload.pop('recipe_id', None)
    return 'recipe-v1:' + sha256(_canonical(payload)).hexdigest()


def validate_recipe(recipe, *, verify_artifacts=True):
    """Validate envelope/causality/bytes; a profile must still admit lowering.

    Return a detached JSON value. No mutable caller object or prior validation
    result is used as a cached proof. Live state/input guards remain runtime
    responsibilities, independent of the retained recipe's byte identity.
    """
    value = _copy(recipe)
    if (type(value) is not dict or set(value) != _FIELDS or value['schema'] != SCHEMA
            or type(value['schema_version']) is not int or value['schema_version'] != VERSION
            or value['recipe_id'] != recipe_id(value)
            or type(value['profile']) is not str or not value['profile']):
        raise ValueError('execution recipe schema, fields or identity differs')
    for field in ('semantic_boundary', 'control', 'resources', 'implementation'):
        if type(value[field]) is not dict or not value[field]:
            raise ValueError('explicit recipe ' + field + ' required')
    boundary = value['semantic_boundary']
    if type(boundary.get('id')) is not str or not boundary['id']:
        raise ValueError('semantic boundary identity required')
    artifacts = value['artifacts']
    if type(artifacts) is not dict or not artifacts:
        raise ValueError('original recipe component references required')
    seen = {}
    for role, reference in artifacts.items():
        if type(role) is not str or not role: raise ValueError('component role required')
        reference = _reference(reference)
        if reference['uri'] in seen and seen[reference['uri']] != reference:
            raise ValueError('one component path has conflicting identities')
        if verify_artifacts and reference['uri'] not in seen:
            read_artifact(reference, json_document=False)
        seen[reference['uri']] = reference
    implementation = value['implementation']
    if type(implementation.get('family')) is not str or not implementation['family']:
        raise ValueError('explicit backend implementation family required')
    inputs = implementation.get('backend_inputs')
    if (type(inputs) is not dict or not inputs or any(type(name) is not str or not name
            or type(role) is not str or role not in artifacts for name, role in inputs.items())):
        raise ValueError('backend inputs must reference original component roles')
    from scar.optimization.schedule import schedule_from_dict
    schedule_from_dict(value['task_graph'])
    obligations = value['obligations']
    if type(obligations) is not list: raise ValueError('explicit obligation ledger required')
    identifiers = set()
    for item in obligations:
        if (type(item) is not dict or type(item.get('id')) is not str or not item['id']
                or item['id'] in identifiers or item.get('status') not in _OBLIGATION_STATES
                or type(item.get('detail')) is not str or not item['detail']):
            raise ValueError('unique typed recipe obligations required')
        identifiers.add(item['id'])
    return value


def make_recipe(profile, *, semantic_boundary, control, task_graph, resources,
                implementation, artifacts, obligations=()):
    from scar.optimization.schedule import ResourceSchedule, schedule_document
    if isinstance(task_graph, ResourceSchedule): task_graph = schedule_document(task_graph)
    value = dict(schema=SCHEMA, schema_version=VERSION, profile=profile,
                 semantic_boundary=semantic_boundary, control=control, task_graph=task_graph,
                 resources=resources, implementation=implementation,
                 artifacts={role:_reference(ref) for role, ref in artifacts.items()},
                 obligations=list(obligations))
    value = _copy(value); value['recipe_id'] = recipe_id(value)
    return validate_recipe(value)


def read_recipe(path, *, verify_artifacts=True):
    return validate_recipe(_json(_bytes(path)), verify_artifacts=verify_artifacts)


def write_recipe(recipe, path):
    value = validate_recipe(recipe); path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n')
    return artifact_reference(path, SCHEMA)


__all__ = ['SCHEMA', 'VERSION', 'make_recipe', 'validate_recipe', 'recipe_id',
           'artifact_reference', 'open_artifact', 'read_artifact', 'read_recipe', 'write_recipe']

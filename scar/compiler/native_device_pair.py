"""Admit one native CPU entry and one CUDA trial under a shared file ABI."""

from hashlib import sha256
import json
from math import prod
import re
import sys

from scar.compiler.compile_exported_aten import SCHEMA as CPU_SCHEMA, _render_launch
from scar.compiler.compile_exported_aten_cuda import SCHEMA as CUDA_SCHEMA
from scar.ir.v2.schemas import canonical_json
from scar.runtime.validate_native_project import verify_native_project


SCHEMA = 'scar.native-device-pair'


def _identity(record):
    digest, size = record['sha256'], record['size_bytes']
    if type(digest) is not str or re.fullmatch('[0-9a-f]{64}', digest) is None or type(size) is not int or size < 0:
        raise ValueError('device pair requires complete content identities')
    return dict(sha256=digest, size_bytes=size)


def _digest(raw):
    return dict(sha256=sha256(raw).hexdigest(), size_bytes=len(raw))


def _manifest(root, digest, schema):
    raw = (root / 'artifact.json').read_bytes()
    manifest = json.loads(raw)
    if sha256(raw).hexdigest() != digest:
        raise ValueError('native project manifest changed during pair admission')
    if manifest['schema'] != schema or manifest['schema_version'] != 1:
        raise ValueError('device pair requires a native export of the expected implementation kind')
    return manifest


def _member(root, manifest, name):
    path = root / name
    if name not in manifest['files'] or path.resolve(strict=True) != path or not path.is_relative_to(root):
        raise ValueError(f'pair admission requires a locked local member: {name}')
    raw = path.read_bytes()
    if _digest(raw) != _identity(manifest['files'][name]):
        raise ValueError(f'native project member changed during pair admission: {name}')
    return raw


def _ports(rows, *, cuda=False):
    ports = []
    for index, row in enumerate(rows):
        kind, shape = row['kind'], row['shape']
        if (kind not in ('INPUT', 'WEIGHT') or type(shape) is not list or len(shape) != 2
                or any(type(extent) is not int or extent <= 0 for extent in shape)):
            raise ValueError('device pair requires ordinary rank-two FP32 file inputs and weights')
        size = 4 * prod(shape)
        port = dict(ordinal=index, id=row['id'], kind=kind, shape=shape, size_bytes=size)
        if kind == 'WEIGHT':
            payload = _identity(row['payload'] if cuda else row)
            if payload['size_bytes'] != size:
                raise ValueError('packaged weight byte extent differs from its ordered pair port')
            port['weight_payload'] = payload
        if cuda and row['nbytes'] != size:
            raise ValueError('CUDA input byte extent differs from its projected shape')
        ports.append(port)
    return ports


def _selection(root, manifest, cpu_profile, cuda_execution):
    if 'profiles' not in manifest:
        if cpu_profile is not None:
            raise ValueError('a single CPU specialization has no dispatch label')
        choices = {None: 'plan'}
    else:
        if cpu_profile is not None and cpu_profile not in manifest['profiles']:
            raise ValueError('requested CPU specialization is not published')
        choices = {label: f'plan/profiles/{label}' for label in manifest['profiles']}
    executions = {label: json.loads(_member(root, manifest, f'{base}/execution.json'))
                  for label, base in choices.items()}
    matched = [label for label, plan in executions.items() if (cpu_profile is None or label == cpu_profile)
               and plan['symbolic_profile'] == cuda_execution['symbolic_profile']
               and _identity(plan['program_graph']) == _identity(cuda_execution['program'])]
    if len(matched) != 1:
        raise ValueError('expected one compatible CPU specialization with the same typed ProgramGraph and shape assignment')
    selected, = matched
    execution = executions[selected]
    if (execution['schema'] != 'scar.native-tensor-entry-plan' or execution['schema_version'] != 1
            or execution['source'] != manifest['source']):
        raise ValueError('CPU specialization differs from the native export entry and source binding')
    launch = _render_launch(execution['inputs'], len(execution['output_shapes']),
                            {label: plan['cpu_ids'] for label, plan in executions.items()})
    if _member(root, manifest, 'launch.sh') != launch.encode():
        raise ValueError('CPU launcher differs from the qualified ordered file ABI')
    return selected, choices[selected], execution


def _package(manifest, digest):
    executable = manifest['executable']
    return dict(schema=manifest['schema'], status=manifest['status'], manifest_sha256=digest,
                executable=dict(path=executable, **_identity(manifest['files'][executable])),
                launcher=dict(path='launch.sh', **_identity(manifest['files']['launch.sh'])))


def admit_native_device_pair(cpu_project, cuda_project, *, cpu_manifest_sha256, cuda_manifest_sha256,
                             cpu_profile=None):
    """Return verified local roots and a content-addressed comparison witness."""
    cpu_root, cpu_receipt = verify_native_project(cpu_project, cpu_manifest_sha256)
    cuda_root, cuda_receipt = verify_native_project(cuda_project, cuda_manifest_sha256)
    cpu = _manifest(cpu_root, cpu_manifest_sha256, CPU_SCHEMA)
    cuda = _manifest(cuda_root, cuda_manifest_sha256, CUDA_SCHEMA)
    if (cpu_receipt['native_dependencies_checked'] is not True or cuda_receipt['native_dependencies_checked'] is not True
            or cuda_receipt.get('offline_schedule_replayed') is not True):
        raise ValueError('device pair requires verified native export projects and a replayed CUDA plan')
    source = _identity(cpu['source'])
    if source != _identity(cuda['source']):
        raise ValueError('CPU and CUDA projects come from different exported source contents')
    cuda_execution = json.loads(_member(cuda_root, cuda, 'plan/execution.json'))
    label, base, cpu_execution = _selection(cpu_root, cpu, cpu_profile, cuda_execution)
    recipe_name = 'source-recipe.json' if cpu.get('simplified_normal_completion') else 'recipe.json'
    cpu_recipe = json.loads(_member(cpu_root, cpu, f'{base}/{recipe_name}'))
    cuda_recipe = json.loads(_member(cuda_root, cuda, 'plan/recipe.json'))
    if cpu_recipe != cuda_recipe:
        raise ValueError('CPU and CUDA projects do not share the original normalized recipe')
    inputs = _ports(cpu_execution['inputs'])
    if inputs != _ports(cuda_execution['inputs'], cuda=True):
        raise ValueError('CPU and CUDA projects do not share ordered file input and weight content identities')
    projection = json.loads(_member(cuda_root, cuda, 'plan/projection.json'))
    shapes = {op['id']: op['shape'] for op in projection['operations']}
    outputs = [dict(ordinal=index, id=name, shape=shapes[name], size_bytes=4 * prod(shapes[name]))
               for index, name in enumerate(cuda_recipe['outputs'])]
    if [row['shape'] for row in outputs] != cpu_execution['output_shapes']:
        raise ValueError('CPU and CUDA projects do not share ordered output file extents')
    bindings = json.loads(_member(cuda_root, cuda, 'plan/bindings.json'))
    witness = dict(schema=SCHEMA, schema_version=1, cpu_profile=label, source=source,
        program=_identity(cuda_execution['program']), symbolic_profile=cuda_execution['symbolic_profile'],
        original_recipe_sha256=sha256(canonical_json(cuda_recipe).encode()).hexdigest(), inputs=inputs, outputs=outputs,
        abi=dict(version=1, dtype='float32', layout='row_major_contiguous', byte_order=sys.byteorder,
                 observation='ORDERED_OUTPUT_FILE_BYTES'),
        cpu=dict(_package(cpu, cpu_manifest_sha256), cpu_ids=cpu_execution['cpu_ids']),
        cuda=dict(_package(cuda, cuda_manifest_sha256), device=bindings['device']))
    return cpu_root, cuda_root, witness


def replay_native_device_pair(witness, cpu_project, cuda_project):
    """Read the pinned projects again and require the same matched file ABI."""
    if witness.get('schema') != SCHEMA or type(witness.get('schema_version')) is not int or witness['schema_version'] != 1:
        raise ValueError('unknown device pair witness format')
    cpu_root, cuda_root, observed = admit_native_device_pair(cpu_project, cuda_project,
        cpu_manifest_sha256=witness['cpu']['manifest_sha256'], cuda_manifest_sha256=witness['cuda']['manifest_sha256'],
        cpu_profile=witness['cpu_profile'])
    if observed != witness:
        raise ValueError('device pair witness differs from the replayed project and file ABI identities')
    return cpu_root, cuda_root, observed

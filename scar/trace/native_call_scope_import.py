"""Bind closed CPython CALL scopes to same-capture provider observations."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path, PurePosixPath
import re

from scar.ir.execution.coordinates import HostThreadInstance
from scar.ir.program.call_edges import CapturedSite, InvocationKey
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import decode, encode, loads
from scar.trace.capture_coordinates import collector_namespace
from scar.trace.reader import open_binary
from scar.trace.torch_call_scopes import LIMIT, PROFILE


SCHEMA = 'scar.native-call-scope-observations'
_MAX_LINE = 2_000_000


def _require(condition, detail):
    if not condition:
        raise ValueError('native CALL scope: ' + detail)


def _positive(value):
    return type(value) is int and value > 0


def _member(coordinates, logical, schema, pid):
    matches = [row for row in coordinates.report['membership']
               if row['logical_path'] == str(logical)]
    _require(len(matches) == 1, 'no unique exact capture member for ' + str(logical))
    row, = matches
    artifact = decode(ProviderArtifact, row['artifact'])
    _require(artifact.uri == str(logical) and artifact.schema == schema
             and coordinates.process_for(artifact, pid) == coordinates.target_process,
             'artifact has a different schema or capture process')
    return artifact, Path(row.get('local_path', artifact.uri))


def _select_rows(path, artifact, pid, selector, wanted, budget):
    """Stream original bytes and retain only indexed events used by the join."""
    selected, digest, previous, size = {}, sha256(), 0, 0
    with open_binary(path) as stream:
        for number in range(1, budget['lines'] + 2):
            allowance = min(_MAX_LINE, budget['bytes'] - size)
            raw = stream.readline(allowance + 1)
            if not raw:
                break
            _require(len(raw) <= allowance and number <= budget['lines'],
                     'original CALL stream exceeds the configured scan budget')
            size += len(raw)
            digest.update(raw)
            row = loads(raw.decode('utf-8'))
            _require(type(row) is dict and type(row.get('process_id')) is int and row['process_id'] == pid
                     and _positive(row.get('index')) and row['index'] > previous,
                     'original CALL stream changed process or event index order')
            previous = row['index']
            key = selector(row)
            if key in wanted:
                _require(key not in selected, 'original CALL event was repeated')
                selected[key] = row, ProviderFactReference(artifact, '#/lines/' + str(number))
        else:
            raise ValueError('native CALL scope: original stream scan did not reach EOF')
    _require((size, digest.hexdigest()) == (artifact.size_bytes, artifact.sha256),
             'original CALL stream changed after capture membership validation')
    budget['bytes'] -= size
    budget['lines'] -= number - 1
    return selected


def _reference(artifact, pointer, array):
    prefix = 'sha256:' + artifact.sha256
    _require(type(pointer) is str and pointer.startswith(prefix)
             and re.fullmatch(r'#/' + array + r'/[0-9]+', pointer[len(prefix):]) is not None,
             'native instance does not refer to the exact original provider artifact')
    return ProviderFactReference(artifact, pointer[len(prefix):])


def _rf_member(instance, *, pid, native_tid, category, eto, kineto):
    metadata = instance.metadata
    rf = metadata.get('rf_id')
    if metadata.get('kineto_reference') is None or not _positive(rf):
        return None
    _require(metadata.get('category') == category and metadata.get('raw_pid') == pid
             and metadata.get('raw_tid') == native_tid and instance.thread_id == native_tid
             and _positive(metadata.get('eto_tid')),
             'RF-correlated provider observation has a different host actor or category')
    return dict(instance=instance.id.as_dict(), provider_node_id=metadata['node_id'], record_function_id=rf,
                eto=encode(_reference(eto, metadata.get('raw_reference'), 'nodes')),
                kineto=encode(_reference(kineto, metadata['kineto_reference'], 'traceEvents')))


def _configuration(manifest):
    config = manifest.get('configuration', {}).get('native_call_scopes')
    _require(type(config) is dict and config.get('profile') == PROFILE,
             'unsupported collector declaration')
    rows = config.get('rows')
    limit, exhausted, errors = config.get('limit'), config.get('budget_exhausted'), config.get('errors')
    _require(type(rows) is list and _positive(limit) and len(rows) <= limit <= LIMIT
             and type(exhausted) is bool and (not exhausted or len(rows) == limit)
             and type(errors) is list and len(errors) <= LIMIT
             and all(type(error) is str and 0 < len(error) <= 128 for error in errors),
             'malformed collector budget or scope rows')
    if rows:
        _require(_positive(config.get('native_tid')) and _positive(config.get('owner_python_ident'))
                 and config['owner_python_ident'] == manifest.get('install_thread_id'),
                 'collected scopes need their original host and interpreter thread')
    attempts = set()
    for row in rows:
        _require(type(row) is dict and set(row) == {'attempt', 'invocation', 'site', 'outcome'}
                 and _positive(row['attempt']) and row['attempt'] not in attempts
                 and _positive(row['invocation']) and row['outcome'] in ('C_RETURN', 'C_RAISE', 'UNCLOSED'),
                 'invalid or repeated closed scope identity')
        site = decode(CapturedSite, row['site'])
        _require(re.fullmatch(r'pycode-v2:[0-9a-f]{64}', site.code) is not None and site.offset >= 0,
                 'invalid original CALL CodeID or bytecode offset')
        attempts.add(row['attempt'])
    return config, attempts


def import_native_call_scopes(result, manifest_path, coordinates, *, max_bytes, max_lines):
    """Attach provenance to ETO marker occurrences before native graph publication.

    The marker's children already have ETO collector-stack parentage. This
    binds the marker occurrence to a CPython CALL attempt and preserves each
    qualified immediate child. The graph gains no semantic control edges.
    """
    local_manifest = Path(manifest_path).absolute()
    manifest_name = PurePosixPath(coordinates.report['original_root']) / 'trace' / local_manifest.name
    pid = coordinates.target_process.pid
    manifest_artifact, bound_manifest_path = _member(coordinates, manifest_name,
        'scar.execution-evidence-collection', pid)
    _require(local_manifest == bound_manifest_path and manifest_artifact.size_bytes <= 2_000_000,
             'native manifest path or size differs')
    payload = local_manifest.read_bytes()
    _require((len(payload), sha256(payload).hexdigest()) == (manifest_artifact.size_bytes, manifest_artifact.sha256),
             'native manifest changed after capture membership validation')
    manifest = loads(payload.decode('utf-8'))
    _, nonce, collected = collector_namespace(manifest)
    _require(collected and result.report['pid'] == manifest['pid'] == pid
             and result.report['provider_revision'] == manifest['provider']['torch_git_version'],
             'native graph and collector manifest have a different provider or process')
    config, attempts = _configuration(manifest)
    roots = {}
    for key, field, schema in (
        ('eto', 'execution_trace', 'pytorch.execution-trace.v1'),
        ('kineto', 'kineto', 'pytorch.kineto.chrome-trace.v1'),
    ):
        logical = manifest['artifacts'][field]['path']
        roots[key], local = _member(coordinates, logical, schema, pid)
        recorded = result.report['artifacts'][key]
        _require((recorded['path'], recorded['sha256'], recorded['size_bytes'])
                 == (str(local), roots[key].sha256, roots[key].size_bytes),
                 'native graph was imported from a different artifact')
    eto, kineto = roots['eto'], roots['kineto']
    native_tid, python_ident = config.get('native_tid'), config.get('owner_python_ident')
    actor = coordinates.thread_for(kineto, native_tid, 'linux_native_tid')

    operand_rows, frames = {}, {}
    if attempts:
        operand, operand_path = _member(coordinates, manifest_name.parent / f'call-operands.{pid}.jsonl',
                                        'scar.call-operands.events.v1', pid)
        frame, frame_path = _member(coordinates, manifest_name.parent / f'call-events.{pid}.jsonl',
                                    'scar.call-edge-events.v1', pid)
        budget = {'bytes': max_bytes, 'lines': max_lines}
        operand_rows = _select_rows(operand_path, operand, pid, lambda row: row['index'], attempts, budget)
        invocation_ids = {row['invocation'] for row in config['rows']}
        frames = _select_rows(frame_path, frame, pid, lambda row: (row.get('kind'), row.get('invocation_id')),
            {(kind, invocation) for kind in ('call_enter', 'call_exit') for invocation in invocation_ids}, budget)
        _require(set(operand_rows) == attempts and {i for k, i in frames if k == 'call_enter'} == invocation_ids,
                 'scope does not address retained operand attempts and caller entries')

    definitions = result.bundle.semantic.definitions
    children, markers = defaultdict(list), {}
    prefix = f'scar-native-call:{nonce}:{pid}:'
    for instance in result.bundle.evidence.instances.values():
        if instance.metadata.get('provider') != 'ETO':
            continue
        definition = definitions[instance.definition]
        if definition.label.startswith(prefix):
            _require(definition.label not in markers, 'duplicate original ETO call scope marker')
            markers[definition.label] = instance
        if definition.metadata.get('op_schema'):
            children[instance.parent].append((instance, definition))

    gaps, observations, attachments = Counter(), [], []
    for ordinal, row in enumerate(config['rows']):
        attempt, invocation = row['attempt'], row['invocation']
        original, operand_ref = operand_rows[attempt]
        owner, frame_ref = frames['call_enter', invocation]
        exit_row = frames.get(('call_exit', invocation))
        _require(original.get('kind') == 'CALL_ATTEMPT' and original.get('site') == row['site']
                 and original.get('invocation_id') == invocation and original.get('thread_id') == python_ident
                 and owner.get('thread_id') == python_ident and owner.get('code') == row['site']['code']
                 and type(original.get('frame_index')) is int and original['frame_index'] >= owner['index']
                 and (exit_row is None or original['frame_index'] < exit_row[0]['index']),
                 'closed scope does not match the retained original CALL/owner/site')
        observation = dict(attempt=attempt, invocation=encode(InvocationKey(pid, python_ident, invocation)),
            site=row['site'], outcome=row['outcome'], operand=encode(operand_ref), owner=encode(frame_ref),
            declaration=encode(ProviderFactReference(manifest_artifact, f'#/configuration/native_call_scopes/rows/{ordinal}')),
            operators=[], marker=None)
        observations.append(observation)
        marker = markers.get(prefix + str(attempt))
        if config['errors']:
            reason = 'COLLECTOR_DECLARED_SCOPE_ERRORS'
        elif row['outcome'] == 'UNCLOSED':
            reason = 'SCOPE_NOT_CLOSED_BY_ORIGINAL_C_EVENT'
        elif type(actor) is not HostThreadInstance or actor.process != coordinates.target_process.id:
            reason = 'HOST_THREAD_MEMBERSHIP_UNVERIFIED'
        elif marker is None:
            reason = 'ETO_SCOPE_MARKER_NOT_RETAINED'
        else:
            pair = _rf_member(marker, pid=pid, native_tid=native_tid, category='user_annotation', eto=eto, kineto=kineto)
            reason = 'MARKER_RF_UNVERIFIED' if pair is None else None
        if reason is not None:
            observation['status'] = reason
            gaps[reason] += 1
            continue
        observation.update(status='CLOSED_SCOPE_OBSERVED', marker=pair)
        for child, definition in children.get(marker.id, ()):
            op = _rf_member(child, pid=pid, native_tid=native_tid, category='cpu_op', eto=eto, kineto=kineto)
            if op is None:
                gaps['DIRECT_SCHEMA_CHILD_RF_UNVERIFIED'] += 1
                continue
            _require(child.metadata['eto_tid'] == marker.metadata['eto_tid'],
                     'direct ETO schema child has a different provider thread')
            observation['operators'].append(dict(op, label=definition.label, schema=definition.metadata['op_schema']))
        attachments.append((marker, {key: observation[key] for key in ('attempt', 'invocation', 'site', 'outcome', 'operand', 'owner', 'declaration')}))

    for marker, attachment in attachments:
        marker.metadata['native_call_scope_observation'] = attachment
    return dict(schema=SCHEMA, schema_version=1, profile=PROFILE, run=encode(coordinates.run.id),
        process=encode(coordinates.target_process.id), native_tid=native_tid,
        manifest=encode(manifest_artifact), observations=observations, gaps=dict(gaps),
        scope_rows=len(observations), closed_joined_scopes=len(attachments),
        direct_schema_operator_observations=sum(len(row['operators']) for row in observations),
        collector_budget_exhausted=config['budget_exhausted'], collector_errors=config['errors'],
        collector_availability=config.get('status'), clock_matching_used=False,
        relation='CPYTHON_NATIVE_CALL_SCOPE_TO_DIRECT_ETO_SCHEMA_CHILDREN',
        all_children_covered=False, canonical_source_route_established=False,
        operand_routing_established=False, resource_readiness_established=False,
        effects_established=False, transformation_authorized=False)

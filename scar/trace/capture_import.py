"""Capture import."""

from __future__ import annotations

import gc
import gzip
import io
import json
import time
import base64
import hashlib
import os
import re
import sys
import tokenize
from hashlib import sha256
from pathlib import Path
from scar.ir.execution.common import EvidencePoint
from scar.ir.execution.resources import EndedAt
from scar.ir.record_codec import encode


# Assemble existing evidence adapters in one verified capture coordinate space.
#
# This is an execution-evidence assembly, not a substitute ProgramSnapshot.
# Canonical source/control/value/effect/ready gaps remain explicit.  Source,
# Torch and OS adapters retain their own fact IDs; only proved execution
# coordinates are shared before any output snapshot is hashed.
def _write(path, document):
    """Serialize metadata without one gzip write per JSON punctuation token."""
    path = Path(path)
    with path.open('xb') as raw:
        compressed = gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) if path.suffix == '.gz' else raw
        try:
            with io.TextIOWrapper(compressed, encoding='utf-8', write_through=False) as text:
                json.dump(document, text, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
                text.write('\n')
        finally:
            if compressed is not raw:
                compressed.close()
    return path


def _stored(path):
    with Path(path).open('rb') as stream:
        digest = sha256()
        while block := stream.read(1_048_576):
            digest.update(block)
    return {'path': str(Path(path).resolve()), 'stored_sha256': digest.hexdigest(),
            'stored_bytes': Path(path).stat().st_size}


def _check_native_coordinates(document, coordinates):
    expected_run, process = encode(coordinates.run), encode(coordinates.target_process)
    if document['runs'] != [expected_run] or document['processes'] != [process]:
        raise ValueError('native adapter did not use the exact bound capture run/process')
    count = 0
    for row in document['target_lifetime_events']:
        point = row['at']
        if point['run'] != expected_run['id'] or point['process'] != process['id']:
            raise ValueError('native lifetime escaped shared capture coordinates')
        count += 1
    return count


def _check_os_coordinates(fragment, coordinates):
    if fragment.run != coordinates.run:
        raise ValueError('OS adapter changed the shared run declaration')
    processes = {process.id: process for process in coordinates.processes}
    threads = {thread.id: thread for thread in coordinates.threads}
    for process in fragment.processes:
        if processes.get(process.id) != process:
            raise ValueError('OS process does not equal the bound process declaration')
    for thread in fragment.threads:
        if threads.get(thread.id) != thread:
            raise ValueError('OS thread does not equal the bound thread declaration')
    points = []
    points.extend(row.at for row in fragment.target_lifetime_events)
    for row in fragment.handles:
        points.append(row.start)
        if type(row.end) is EndedAt:
            points.append(row.end.point)
    missing = 0
    for point in points:
        if type(point) is not EvidencePoint or point.run != coordinates.run.id:
            raise ValueError('OS evidence point escaped shared run coordinates')
        if point.process is not None and point.process not in processes:
            raise ValueError('OS evidence point uses an unbound process')
        missing += point.process is None
    return {'points_checked': len(points), 'points_with_unresolved_process': missing}


def import_capture(capture_dir, output_dir, *, max_bytes=1_000_000_000, max_lines=2_000_000,
                   schema_contracts=False, bundle_partitions=False, os_interface_work=False,
                   progress=None, components=None):
    """Import an unmodified successful collector capture, without target execution.

    The membership binder validates the original artifacts once.  The existing
    native and OS adapters independently validate their own provider contracts.
    Stored outputs include both the identity joins and their unresolved edges.
    """
    from scar.trace.control_transfers import bind_capture_coordinates
    from scar.trace.native_import import import_native_evidence
    from scar.trace.native_memory import export_native_memory
    from scar.trace.os_resources import import_os_resources, publish_os_interfaces

    if type(os_interface_work) is not bool:
        raise TypeError('os_interface_work must be an exact bool')
    if components not in (None, ('native',), ('native','os')):
        raise ValueError('components must explicitly select native or native+os')
    if components==('native',) and os_interface_work:
        raise ValueError('native-only import cannot claim requested OS interface work')

    capture, out = Path(capture_dir).resolve(), Path(output_dir).resolve()
    if out.exists():
        raise ValueError('capture import requires a fresh output directory')
    out.mkdir(parents=True)
    started, stages, outputs = time.perf_counter(), [], []

    def stage(name, status, **detail):
        record = {'stage': name, 'status': status, 'elapsed_s': time.perf_counter() - started, **detail}
        stages.append(record)
        payload = {'schema': 'scar.capture-import-progress', 'schema_version': 1,
                   'stages': stages, 'whole_model_established': False}
        temporary = out / '.progress.tmp'
        temporary.write_text(json.dumps(payload, indent=2) + '\n')
        temporary.replace(out / 'progress.json')
        if progress is not None:
            progress(record)

    try:
        stage('coordinates', 'RUNNING')
        coordinates = bind_capture_coordinates(capture, max_bytes=max_bytes, max_lines=max_lines,
            profile_uri=str(out / 'capture-coordinates.json'), publish_os_profile=True)
        (out / 'capture-coordinates.json').write_bytes(coordinates.payload)
        outputs.append(_stored(out / 'capture-coordinates.json'))
        stage('coordinates', 'IMPORTED', **coordinates.report['counts'])
        manifests = list((capture / 'trace').glob('execution-evidence.*.json'))
        if len(manifests) != 1:
            raise ValueError('capture manifest became ambiguous after membership validation')
        manifest = json.loads(manifests[0].read_bytes())
        members = coordinates.report['membership']
        def member_path(logical):
            matches = [row.get('local_path', row['artifact']['uri']) for row in members if row['logical_path'] == logical]
            if len(matches) != 1:
                raise ValueError('member has no exact bound artifact: ' + logical)
            return matches[0]
        def native_path(field):
            return member_path(manifest['artifacts'][field]['path'])
        native_out = out / 'native'
        native_out.mkdir()
        stage('native', 'RUNNING')
        native = import_native_evidence(native_path('execution_trace'), native_path('kineto'),
            max_bytes=max_bytes, provider_revision=manifest['provider']['torch_git_version'])
        scope_config = manifest.get('configuration', {}).get('native_call_scopes')
        if scope_config is not None:
            from scar.trace.native_import import import_native_call_scopes
            stage('native_call_scopes', 'RUNNING')
            scopes = import_native_call_scopes(native, manifests[0], coordinates,
                                              max_bytes=max_bytes, max_lines=max_lines)
            scope_path = _write(native_out / 'native-call-scopes.json.gz', scopes)
            native.report['native_call_scopes'] = {
                **{key: value for key, value in scopes.items() if key != 'observations'},
                'artifact': _stored(scope_path),
            }
            stage('native_call_scopes', 'IMPORTED_WITH_OPEN_OBLIGATIONS',
                  scope_rows=scopes['scope_rows'], closed_joined_scopes=scopes['closed_joined_scopes'],
                  direct_schema_operator_observations=scopes['direct_schema_operator_observations'], gaps=scopes['gaps'])
        dispatch_config = manifest.get('configuration', {}).get('dispatch_observations')
        if dispatch_config is not None and dispatch_config.get('started') is True:
            from scar.trace.dispatch_evidence import import_dispatch_bridge
            original_trace = str(Path(manifest['artifacts']['kineto']['path']).parent)
            pid = manifest['pid']
            stage('dispatch_bridge', 'RUNNING')
            import_dispatch_bridge(native,
                member_path(dispatch_config['events']), member_path(dispatch_config['metadata']),
                member_path(original_trace + f'/call-events.{pid}.jsonl'),
                member_path(original_trace + f'/call-codes.{pid}.jsonl'),
                member_path(original_trace + f'/call-meta.{pid}.json'),
                native_manifest_path=member_path(original_trace + '/' + manifests[0].name),
                coordinates=coordinates, max_bytes=max_bytes, max_events=max_lines)
            bridge = native.report['dispatch_bridge']
            _write(native_out / 'dispatch-bridge.json.gz', bridge)
            stage('dispatch_bridge', 'IMPORTED', dispatches=bridge['dispatches'],
                  slot_associations=bridge['slot_associations'], gaps=bridge['gaps'])
        physical_config = manifest.get('configuration', {}).get('tensor_physical_observations')
        if physical_config is not None and physical_config.get('requested') is True:
            from scar.trace.tensor_physical_import import import_tensor_physical
            stage('tensor_physical', 'RUNNING')
            physical = import_tensor_physical(native, member_path(physical_config['events']),
                member_path(physical_config['metadata']), coordinates=coordinates, max_bytes=max_bytes,
                max_rows=max_lines)
            _write(native_out / 'tensor-physical.json.gz', physical)
            stage('tensor_physical', 'IMPORTED_WITH_OPEN_OBLIGATIONS',
                marker_nodes=physical.get('marker_nodes', 0),
                physical_slots=physical.get('physical_slot_associations', 0),
                allocation_epochs=physical.get('allocation_epochs', 0), gaps=physical['gaps'])
        if schema_contracts:
            from scar.trace.native_schema import recover_schema_contracts
            stage('native_schema', 'RUNNING')
            schema = recover_schema_contracts(native, provider_revision=manifest['provider']['torch_git_version'])
            _write(native_out / 'schema-contracts.json.gz', schema)
            stage('native_schema', 'IMPORTED', schemas=schema['unique_schemas_parsed'],
                  formal_slots=schema['formal_slots'], actual_tensor_slot_counts=schema['actual_tensor_slot_counts'])
        memory = export_native_memory(native, native_out, coordinates=coordinates)
        checked_lifetimes = _check_native_coordinates(memory, coordinates) if memory is not None else 0
        if memory is not None:
            _write(native_out / 'native-memory-values.json.gz', native.bundle.values.to_dict())
            _write(native_out / 'native-memory-lifetimes.json.gz', memory)
        if bundle_partitions:
            from scar.store.bundle_partitions import publish_bundle
            stage('bundle_partitions', 'RUNNING')
            partition_manifest = publish_bundle(native.bundle, native_out / 'bundle-partitions',
                progress=lambda record: stage('bundle_partitions', 'WRITING', detail=record))
            stage('bundle_partitions', 'PUBLISHED_WITH_OPEN_OBLIGATIONS')
        else:
            _write(native_out / 'evidence-bundle.json.gz', native.bundle.to_dict())
        _write(native_out / 'provider-references.json.gz', native.references)
        _write(native_out / 'resource-observations.json.gz', native.resources)
        _write(native_out / 'import-report.json.gz', native.report)
        native_summary = {key: value for key, value in native.report.items()
                          if key not in ('memory', 'artifacts', 'dispatch_bridge', 'native_schema', 'tensor_physical') and not isinstance(value, list)}
        if 'dispatch_bridge' in native.report:
            native_summary['dispatch_bridge'] = {key: value for key, value in native.report['dispatch_bridge'].items()
                                                if key not in ('observations', 'paths', 'artifacts')}
        if 'native_schema' in native.report:
            native_summary['native_schema'] = {key: value for key, value in native.report['native_schema'].items()
                                              if key not in ('bindings', 'unqualified_without_schema')}
        if 'tensor_physical' in native.report:
            native_summary['tensor_physical'] = {key: value for key, value in native.report['tensor_physical'].items()
                                                 if key not in ('associations', 'epochs')}
        native_summary['memory'] = {key: value for key, value in native.report.get('memory', {}).items()
                                    if key not in ('records', 'source_artifact')}
        native_summary['shared_lifetime_points_checked'] = checked_lifetimes
        outputs.extend(_stored(path) for path in sorted(native_out.rglob('*')) if path.is_file())
        stage('native', 'IMPORTED', cpu_rf_matches=native.report['cpu_rf_matches'],
              runtime_gpu_matches=native.report['runtime_gpu_matches'], lifetime_points=checked_lifetimes)
        del native, memory
        gc.collect()
        os_summary = None
        if components==('native',):
            stage('os_resources','NOT_REQUESTED_EXPLICIT_COMPONENT_SCOPE')
        else:
            stage('os_resources', 'RUNNING')
            os_out = out / 'os'
            os_out.mkdir()
            os_fragment = import_os_resources(capture / 'os', scope='capture-import',
                root_tid=manifest['pid'], max_bytes=max_bytes, max_lines=max_lines, coordinates=coordinates,
                interface_work=os_interface_work)
            os_check = _check_os_coordinates(os_fragment, coordinates)
            (os_out / 'os-resource-profile.json').write_bytes(os_fragment.protocol_bytes)
            if os_interface_work:
                stage('os_interfaces', 'RUNNING')
                publish_os_interfaces(os_fragment, os_out / 'interfaces',
                    progress=lambda record: stage('os_interfaces', 'WRITING', detail=record))
                stage('os_interfaces', 'PUBLISHED_WITH_OPEN_OBLIGATIONS')
            else:
                _write(os_out / 'os-resources.json.gz', os_fragment.to_dict())
            _write(os_out / 'summary.json', os_fragment.summary)
            outputs.extend(_stored(path) for path in sorted(os_out.rglob('*')) if path.is_file())
            os_summary = dict(os_fragment.summary, coordinate_check=os_check)
            if os_interface_work:
                # Keep collector work distinguishable without filtering or
                # relabeling any original OS observation as application work.
                from scar.analysis.os_postprocessing_roles import bind_snapshot_postprocessor_roles
                from scar.ir.program.providers import ProviderArtifact
                pid = coordinates.target_process.pid
                expected = tuple(capture / 'trace' / (stem + f'.{pid}.jsonl.snapshot-receipt.json')
                    for stem in ('call-events', 'call-codes', 'call-operands',
                                 'python-call-targets', 'function-environments'))
                declared = []
                for path in expected:
                    if path.is_file():
                        if path.stat().st_size > 1_000_000:
                            raise ValueError('snapshot conversion receipt exceeds metadata ceiling')
                        raw = path.read_bytes()
                        declared.append(ProviderArtifact(str(path), sha256(raw).hexdigest(),
                                                         len(raw), 'application/json'))
                if declared:
                    roles = bind_snapshot_postprocessor_roles(coordinates, os_fragment, tuple(declared)).to_dict()
                    roles['expected_receipts_missing'] = [str(p) for p in expected if not p.is_file()]
                    role_path = os_out / 'snapshot-postprocessor-roles.json'
                    _write(role_path, roles)
                    outputs.append(_stored(role_path))
                    os_summary['snapshot_postprocessor_roles'] = {
                        'artifact': _stored(role_path), 'positive_roles': len(roles['roles']),
                        'frontiers': len(roles['frontiers']),
                        'missing_receipts': len(roles['expected_receipts_missing']),
                        'original_observations_filtered': False,
                        'unclassified_is_application': False}
            stage('os_resources', 'IMPORTED', **os_fragment.summary)
            del os_fragment
            gc.collect()
        required = [
            {'id': 'canonical_program_snapshot', 'status': 'OPEN', 'reason': 'source regions and call interfaces still need whole-program composition'},
            {'id': 'python_torch_occurrence', 'status': 'OPEN', 'reason':
             'owned native CALL scopes cover only observed direct provider children'
             if native_summary.get('native_call_scopes', {}).get('closed_joined_scopes', 0)
             else 'shared capture membership is not an invocation-to-RF mapping'},
            {'id': 'logical_value_provenance', 'status': 'OPEN', 'reason': 'provider object/storage tokens do not prove logical content versions'},
            {'id': 'source_state_effects', 'status': 'OPEN', 'reason': 'observations require source read/write and completion correspondences'},
            {'id': 'resource_lifetime_ready', 'status': 'OPEN', 'reason': 'profiler receipts and OS handles do not close every resource lifetime/ready interface'},
        ]
        report = {'schema': 'scar.capture-model', 'schema_version': 1, 'capture': str(capture),
            'coordinate_binding': encode(coordinates.artifact), 'run': encode(coordinates.run),
            'processes': encode(coordinates.processes), 'threads': encode(coordinates.threads),
            'native': native_summary, 'os': os_summary, 'outputs': outputs,
            'coordinate_identity_consistent': True, 'coverage_obligations': required,
            'whole_model_established': False, 'source_runtime_verified': False,
            'transformation_authorized': False, 'elapsed_s': time.perf_counter() - started}
        if 'os_capture_provider' in coordinates.report:
            report['os_capture_provider'] = coordinates.report['os_capture_provider']
        if components is not None:
            report['component_scope']={'selected':list(components),'skipped':[] if 'os' in components else ['os'],
                'scope':'only requested fragments imported; skipped component does not imply completion'}
        if bundle_partitions:
            report['native_bundle_storage'] = {'kind': 'partitioned',
                'manifest': str(native_out / 'bundle-partitions' / 'manifest.json'),
                'publication_is_model_completeness': False}
        if os_interface_work:
            report['os_interface_storage'] = {'kind': 'partitioned',
                'manifest': str(os_out / 'interfaces' / 'manifest.json'),
                'publication_is_model_completeness': False}
        _write(out / 'capture-model.json', report)
        stage('capture', 'IMPORTED_WITH_OPEN_OBLIGATIONS', coordinate_identity_consistent=True)
        return report
    except BaseException as error:
        stage('capture', 'FAILED', error=type(error).__name__ + ': ' + str(error))
        raise


# Bounded original source bytes observed alongside live code descriptors.
#
# These are capture inputs, not compiler witnesses. Reading a file after a code
# object exists does not establish which bytes its loader read. Ordinary offline
# CodeWitness matching must independently establish any later source association.
PROGRAM_PORTS_PROFILE = 'SCAR_PROGRAM_PORTS_V1'
PROCESS_PORTS_PROFILE = 'SCAR_PROGRAM_PORTS_V2'
PROGRAM_PORTS_PROFILES = (PROGRAM_PORTS_PROFILE, PROCESS_PORTS_PROFILE)
SOURCE_SCHEMA = 'scar.captured-python-sources'
SOURCE_SCOPE = 'PROJECT_ROOT_AND_ACTUAL_PYTHON_DESCENDANTS_V1'
PROCESS_SOURCE_SCOPE = 'TARGET_PROCESS_INSTALLED_PYTHON_THREADS_V1'


def _wire(value):
    return (json.dumps(value, ensure_ascii=True, separators=(',', ':'), allow_nan=False)+'\n').encode('ascii')


def _stamp(value):
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


class CapturedSources:
    def __init__(self, out, pid, *, max_files=1024, max_file_bytes=2_000_000,
                 max_total_bytes=64_000_000, profile=PROGRAM_PORTS_PROFILE):
        if any(type(x) is not int or x < 1 for x in (pid, max_files, max_file_bytes, max_total_bytes)):
            raise ValueError('captured source budgets must be positive integers')
        self.pid, self.out = pid, Path(out)
        if profile not in PROGRAM_PORTS_PROFILES:
            raise ValueError('unsupported captured-source scope profile')
        self.profile = profile
        self.limits = dict(max_files=max_files, max_file_bytes=max_file_bytes, max_total_bytes=max_total_bytes)
        self.path = self.out / f'captured-sources.{pid}.jsonl'
        self.meta = self.out / f'captured-source-meta.{pid}.json'
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.files, self.codes = {}, {}
        self.raw_read = self.written = self.rows = 0
        self.digest = hashlib.sha256()
        self.closed = False

    def _write(self, row):
        raw = _wire(row)
        at = 0
        while at < len(raw):
            n = os.write(self.fd, raw[at:])
            if n <= 0:
                raise OSError('captured source write made no progress')
            self.digest.update(raw[at:at+n]); self.written += n; at += n
        self.rows += 1

    def observe(self, description):
        """One descriptor link per CodeID; one source read per original path."""
        key, filename = description['key'], description['filename']
        if key in self.codes:
            return
        self.codes[key] = filename
        if filename in self.files:
            return
        item = dict(path=filename, status='UNAVAILABLE', frontier=None,
                    compiler_match='NOT_CHECKED', loader_bytes_observed=False)
        self.files[filename] = item
        raw = None
        try:
            if not os.path.isabs(filename):
                item['frontier'] = 'NON_FILESYSTEM_CODE_FILENAME'; return
            if len(self.files) > self.limits['max_files']:
                item['frontier'] = 'SOURCE_FILE_COUNT_BUDGET'; return
            path = Path(filename)
            before = path.stat()
            item['stamp_at_observation'] = _stamp(before)
            if not path.is_file():
                item['frontier'] = 'SOURCE_PATH_NOT_REGULAR_FILE'; return
            if before.st_size > self.limits['max_file_bytes']:
                item['frontier'] = 'SOURCE_FILE_BYTE_BUDGET'; return
            remaining = self.limits['max_total_bytes'] - self.raw_read
            if before.st_size > remaining:
                item['frontier'] = 'SOURCE_TOTAL_BYTE_BUDGET'; return
            # The descriptor stamp and the open file must agree. No re-open or
            # retry can silently replace the first captured source version.
            with path.open('rb') as stream:
                if _stamp(os.fstat(stream.fileno())) != _stamp(before):
                    item['frontier'] = 'SOURCE_CHANGED_BEFORE_READ'; return
                raw = stream.read(min(self.limits['max_file_bytes'], remaining))
                self.raw_read += len(raw)
                if _stamp(os.fstat(stream.fileno())) != _stamp(before):
                    item['frontier'] = 'SOURCE_CHANGED_DURING_READ'; return
            if len(raw) != before.st_size or _stamp(path.stat()) != _stamp(before):
                item['frontier'] = 'SOURCE_CHANGED_DURING_READ'; return
            encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
            text = raw.decode(encoding).replace('\r\n', '\n').replace('\r', '\n')
            normalized = text.encode('utf-8')
            item.update(status='ARCHIVED_SOURCE_BYTES', raw_sha256=hashlib.sha256(raw).hexdigest(),
                raw_bytes=len(raw), encoding=encoding,
                normalized_sha256=hashlib.sha256(normalized).hexdigest(),
                normalized_bytes=len(normalized), observation='FIRST_DESCRIPTOR_OBSERVATION')
            self._write(dict(schema=SOURCE_SCHEMA, schema_version=1, process_id=self.pid,
                index=self.rows+1, **item, raw_base64=base64.b64encode(raw).decode('ascii')))
            item['source_row'] = self.rows
        except (OSError, UnicodeError, SyntaxError, LookupError) as error:
            item['status'] = 'UNAVAILABLE'
            item['frontier'] = 'SOURCE_CAPTURE_' + type(error).__name__

    def close(self):
        if self.closed or os.getpid() != self.pid:
            return
        self.closed = True
        os.close(self.fd)
        for item in self.files.values():
            if item['status'] != 'ARCHIVED_SOURCE_BYTES':
                continue
            try:
                unchanged = _stamp(Path(item['path']).stat()) == item['stamp_at_observation']
            except OSError:
                unchanged = False
            item['unchanged_at_close'] = unchanged
            if not unchanged:
                item['frontier'] = 'SOURCE_CHANGED_AFTER_OBSERVATION'
        doc = dict(schema=SOURCE_SCHEMA, schema_version=1, profile=self.profile,
            process_id=self.pid, limits=self.limits, files=list(self.files.values()),
            code_sources=[dict(code=key, path=path) for key, path in self.codes.items()],
            file=dict(uri=str(self.path), sha256=self.digest.hexdigest(), size_bytes=self.written,
                      rows=self.rows), source_raw_bytes_read=self.raw_read,
            compiler_match='NOT_CHECKED', loader_bytes_observed=False,
            runtime_optimize=sys.flags.optimize,
            compile_flags_and_per_code_optimize='NOT_OBSERVED')
        raw = _wire(doc)
        if len(raw) > 8_000_000:
            raise ValueError('captured source metadata byte budget exceeded')
        with self.meta.open('xb') as stream:
            stream.write(raw)
        return dict(uri=str(self.meta), sha256=hashlib.sha256(raw).hexdigest(),
                    size_bytes=len(raw), schema=SOURCE_SCHEMA+'/1')


def validate_program_profile(calls, operands):
    """Metadata contract only; source bytes and code matching remain separate."""
    from scar.ir.program.call_operands import PROGRAM_PORTS_RULE, PROCESS_PORTS_RULE
    profile = calls.get('capture_profile')
    requested = profile in PROGRAM_PORTS_PROFILES
    expected = PROGRAM_PORTS_RULE if profile == PROGRAM_PORTS_PROFILE else PROCESS_PORTS_RULE
    if requested != (operands.get('rule') in (PROGRAM_PORTS_RULE, PROCESS_PORTS_RULE)) or requested and operands['rule'] != expected:
        raise ValueError('program ports frame/operand profile differs')
    if not requested:
        return None
    scope = SOURCE_SCOPE if profile == PROGRAM_PORTS_PROFILE else PROCESS_SOURCE_SCOPE
    if (calls.get('scope_policy') != scope or calls.get('instruction_monitoring') is not False
            or 'capture_plan' in calls or operands.get('instructions') != 0
            or type(calls.get('captured_sources')) is not dict):
        raise ValueError('program ports scope/source declaration differs')
    if profile == PROCESS_PORTS_PROFILE:
        hooks = calls.get('thread_hooks')
        ids = hooks.get('python_thread_identifiers_at_install') if type(hooks) is dict else None
        if (type(hooks) is not dict or hooks.get('api') != 'threading.setprofile_all_threads'
                or hooks.get('enumeration_api') != 'sys._current_frames'
                or type(ids) is not list or not ids or len(set(ids)) != len(ids)
                or any(type(x) is not int or x < 1 for x in ids)
                or type(hooks.get('python_threads_at_install')) is not int
                or hooks['python_threads_at_install'] != len(ids)
                or hooks.get('installing_python_ident') not in ids
                or type(hooks.get('installing_native_tid')) is not int or hooks['installing_native_tid'] < 1
                or any(type(hooks.get(x)) is not bool for x in
                       ('known_current_hook_present','known_future_threading_hook_present'))
                or hooks.get('existing_other_thread_hook_history') != 'NOT_OBSERVED'
                or hooks.get('other_thread_hook_restoration') != 'NOT_ENUMERABLE; CLOSED_COLLECTOR_NOOP_IF_STILL_INSTALLED'
                or calls.get('scope_frontiers') != ['FUTURE_NON_THREADING_PYTHON_THREADS_NOT_GUARANTEED',
                    'PREINSTALL_ACTIVE_FRAMES_HAVE_NO_RETROACTIVE_ENTRY',
                    'FORK_AND_EXEC_CHILD_PYTHON_NOT_CAPTURED',
                    'PREEXISTING_OTHER_THREAD_HOOK_HISTORY_NOT_OBSERVED']):
            raise ValueError('process ports thread hook contract differs')
        forwarding = calls.get('target_forwarding')
        roots = forwarding.get('roots') if type(forwarding) is dict else None
        if (type(forwarding) is not dict or forwarding.get('kinds') != ['DISPATCH','IMPORT','TRACE']
                or type(roots) is not list
                or any(type(x) is not list or len(x) != 2 or type(x[0]) is not int or x[0] < 1
                       or type(x[1]) is not int or not 0 <= x[1] < 3 for x in roots)
                or len({x[0] for x in roots}) != len(roots)
                or type(forwarding.get('roots_seen')) is not int or forwarding['roots_seen'] < len(roots)
                or forwarding.get('max_metadata_bytes') != 1_000_000
                or sum(len(json.dumps(x,separators=(',',':')))+1 for x in roots) > 1_000_000
                or type(forwarding.get('frontiers')) is not list
                or any(type(x) is not str or not x for x in forwarding['frontiers'])
                or forwarding.get('context') != 'original entry caller site/path/ancestor; not direct-call or submission proof'):
            raise ValueError('process ports target forwarding declaration differs')
    return calls['captured_sources']


def validate_source_metadata(doc, pid, code_sources=None, *, profile=None):
    """Validate provenance metadata without promoting it to compiler evidence."""
    if (set(doc) != {'schema','schema_version','profile','process_id','limits','files','code_sources',
                     'file','source_raw_bytes_read','compiler_match','loader_bytes_observed',
                     'runtime_optimize','compile_flags_and_per_code_optimize'}
            or doc.get('schema') != SOURCE_SCHEMA or type(doc.get('schema_version')) is not int or doc['schema_version'] != 1
            or type(doc.get('process_id')) is not int or doc['process_id'] != pid
            or doc.get('profile') not in PROGRAM_PORTS_PROFILES
            or profile is not None and doc['profile'] != profile
            or doc.get('compiler_match') != 'NOT_CHECKED'
            or doc.get('loader_bytes_observed') is not False
            or doc.get('compile_flags_and_per_code_optimize') != 'NOT_OBSERVED'):
        raise ValueError('captured source schema/issuer/qualification differs')
    limits = doc.get('limits')
    if (type(limits) is not dict or set(limits) != {'max_files','max_file_bytes','max_total_bytes'}
            or any(type(v) is not int or v < 1 for v in limits.values())):
        raise ValueError('captured source limits differ')
    files, links = doc.get('files'), doc.get('code_sources')
    if type(files) is not list or type(links) is not list:
        raise ValueError('captured source inventories missing')
    if any(type(x) is not dict or set(x) != {'code','path'} for x in links):
        raise ValueError('captured source code links contain unexpected fields')
    if any(type(x) is not dict or type(x.get('path')) is not str or not x['path'] for x in files):
        raise ValueError('captured source original path differs')
    by_path = {item['path']: item for item in files}
    actual_codes = {item['code']: item['path'] for item in links}
    if (len(by_path) != len(files) or len(actual_codes) != len(links)
            or any(type(k) is not str or re.fullmatch(r'pycode-v2:[0-9a-f]{64}', k) is None
                   or p not in by_path for k,p in actual_codes.items())
            or code_sources is not None and actual_codes != code_sources):
        raise ValueError('captured source links differ from original code inventory')
    retained = [x for x in files if x['status'] == 'ARCHIVED_SOURCE_BYTES']
    if any(x.get('compiler_match') != 'NOT_CHECKED' or x.get('loader_bytes_observed') is not False
           or x['status'] not in ('ARCHIVED_SOURCE_BYTES','UNAVAILABLE') for x in files):
        raise ValueError('captured source metadata upgrades source qualification')
    if (len(retained) > limits['max_files'] or
            sorted(x['source_row'] for x in retained) != list(range(1,len(retained)+1))
            or any(type(x['raw_bytes']) is not int or not 0 <= x['raw_bytes'] <= limits['max_file_bytes']
                   for x in retained)
            or type(doc['source_raw_bytes_read']) is not int
            or not sum(x['raw_bytes'] for x in retained) <= doc['source_raw_bytes_read'] <= limits['max_total_bytes']):
        raise ValueError('captured source counts/budgets differ')
    declared = doc['file']
    if (type(declared) is not dict or set(declared) != {'uri','sha256','size_bytes','rows'}
            or declared['rows'] != len(retained)
            or type(declared['size_bytes']) is not int or declared['size_bytes'] < 0
            or re.fullmatch(r'[0-9a-f]{64}',declared['sha256']) is None):
        raise ValueError('captured source row receipt differs')
    return by_path


def validate_source_row(row, doc, index, by_path):
    item = by_path.get(row.get('path'))
    if (item is None or item['status'] != 'ARCHIVED_SOURCE_BYTES'
            or row.get('schema') != SOURCE_SCHEMA or row.get('schema_version') != 1
            or row.get('process_id') != doc['process_id'] or row.get('index') != index
            or item['source_row'] != index or row.get('compiler_match') != 'NOT_CHECKED'
            or row.get('loader_bytes_observed') is not False):
        raise ValueError('captured source row identity/qualification differs')
    raw = base64.b64decode(row['raw_base64'], validate=True)
    encoding,_ = tokenize.detect_encoding(io.BytesIO(raw).readline)
    normalized = raw.decode(encoding).replace('\r\n','\n').replace('\r','\n').encode('utf-8')
    for name,value in (('raw_bytes',len(raw)), ('raw_sha256',hashlib.sha256(raw).hexdigest()),
                       ('encoding',encoding), ('normalized_bytes',len(normalized)),
                       ('normalized_sha256',hashlib.sha256(normalized).hexdigest())):
        if row.get(name) != value or item.get(name) != value:
            raise ValueError('captured source original/normalized bytes differ')


__all__ = [
    'import_capture',
    'PROGRAM_PORTS_PROFILE',
    'PROCESS_PORTS_PROFILE',
    'PROGRAM_PORTS_PROFILES',
    'SOURCE_SCHEMA',
    'SOURCE_SCOPE',
    'PROCESS_SOURCE_SCOPE',
    'CapturedSources',
    'validate_program_profile',
    'validate_source_metadata',
    'validate_source_row',
]

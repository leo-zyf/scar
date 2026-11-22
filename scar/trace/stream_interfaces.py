"""Stream interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import inspect
import io
import os
import time
import json
import threading
import types
from hashlib import sha256
from pathlib import Path
from scar.ir.program.call_edges import CapturedSite, InvocationKey
from scar.ir.record_codec import decode, encode


# Point observations of exact builtin stream backing, never a resource graph.
#
# Only saved builtin descriptors are used. A buffered/text receiver can wrap a
# custom raw object, so exact receiver type alone does not authorize fileno().
# The builtin chain must end at exact FileIO; its saved C fileno descriptor is
# called directly. No tell, flush, contents, filename or FD-table lookup occurs.
# FD/fstat observations are non-atomic and do not establish handle lifetime,
# logical contents, cross-call identity, persistence or readiness.
class BackingStatus(str, Enum):
    OBSERVED = 'OBSERVED'
    MEMORY_STREAM = 'MEMORY_STREAM'
    UNSUPPORTED_CHAIN = 'UNSUPPORTED_CHAIN'
    CHAIN_FAILED = 'CHAIN_FAILED'
    FILENO_FAILED = 'FILENO_FAILED'
    FSTAT_FAILED = 'FSTAT_FAILED'


@dataclass(frozen=True, slots=True)
class StreamBackingStat:
    device: int
    inode: int
    mode: int
    rdev: int
    size_bytes: int

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in (self.device,self.inode,self.mode,self.rdev)) or type(self.size_bytes) is not int:
            raise ValueError('stream fstat fields must be their exact integer observations')


@dataclass(frozen=True, slots=True)
class StreamBackingFailure:
    stage: str
    exception_type: str
    errno: int | None

    def __post_init__(self):
        if (self.stage not in ('BACKING_CHAIN','FILENO','FSTAT')
                or type(self.exception_type) is not str or not self.exception_type
                or self.errno is not None and type(self.errno) is not int):
            raise ValueError('stream backing failure must retain its actual stage/type/errno')


@dataclass(frozen=True, slots=True)
class StreamBackingObservation:
    status: BackingStatus
    started_ns: int
    finished_ns: int
    chain_types: tuple[str, ...]
    fd: int | None = None
    stat: StreamBackingStat | None = None
    failure: StreamBackingFailure | None = None
    phase: str = 'CALL_BEFORE_INVOKE'
    clock: str = 'MONOTONIC_NS'

    def __post_init__(self):
        if (type(self.status) is not BackingStatus or self.phase != 'CALL_BEFORE_INVOKE'
                or self.clock != 'MONOTONIC_NS' or type(self.started_ns) is not int or self.started_ns < 0
                or type(self.finished_ns) is not int or self.finished_ns < self.started_ns
                or type(self.chain_types) is not tuple or not 1 <= len(self.chain_types) <= 4
                or any(type(name) is not str or not name for name in self.chain_types)
                or self.fd is not None and (type(self.fd) is not int or self.fd < 0)
                or self.stat is not None and type(self.stat) is not StreamBackingStat
                or self.failure is not None and type(self.failure) is not StreamBackingFailure):
            raise ValueError('invalid builtin stream backing point observation')
        observed = self.status is BackingStatus.OBSERVED
        failed = {BackingStatus.CHAIN_FAILED:'BACKING_CHAIN',BackingStatus.FILENO_FAILED:'FILENO',BackingStatus.FSTAT_FAILED:'FSTAT'}.get(self.status)
        if ((self.stat is not None) != observed
                or (self.fd is not None) != (observed or self.status is BackingStatus.FSTAT_FAILED)
                or (self.failure is not None) != (failed is not None)
                or failed is not None and self.failure.stage != failed):
            raise ValueError('stream backing status does not match its actual fields/failure')


_FILE = io.FileIO
_MEMORY = frozenset((io.BytesIO,io.StringIO))
# Getsets bypass per-instance shadow attributes. Never invoke a raw object's
# possibly overridden fileno, even when the enclosing buffer is exact builtin.
_CHILD = {cls:inspect.getattr_static(cls,'raw') for cls in (io.BufferedReader,io.BufferedWriter,io.BufferedRandom)}
_CHILD[io.TextIOWrapper] = inspect.getattr_static(io.TextIOWrapper,'buffer')
_FILENO = inspect.getattr_static(_FILE,'fileno')
_FSTAT = os.fstat
_NOW = time.monotonic_ns


_TYPE_NAMES={cls:cls.__module__+'.'+cls.__qualname__ for cls in (*_MEMORY,*_CHILD,_FILE,io.BufferedRWPair)}


def _type_name(value):
    # Unknown objects/metaclasses are not introspected; even formatting their
    # type could dispatch user descriptors. The missing chain stays explicit.
    return next((name for cls,name in _TYPE_NAMES.items() if type(value) is cls),'<unsupported-nonbuiltin>')


def observe_stream_backing(receiver):
    """Called only at an original accepted stream CALL observation.

    Local receiver/raw references die before return; no application object is
    stored in the sidecar. Probe failures are evidence about this observation,
    distinct from the program's later C_RETURN/C_RAISE completion.
    """
    started=_NOW();chain=[];current=receiver;seen=set();stage='BACKING_CHAIN';fd=None
    try:
        while True:
            chain.append(_type_name(current))
            cls=type(current)
            if any(cls is known for known in _MEMORY):
                return StreamBackingObservation(BackingStatus.MEMORY_STREAM,started,_NOW(),tuple(chain))
            if cls is _FILE:
                break
            descriptor=next((item for known,item in _CHILD.items() if cls is known),None)
            if descriptor is None or id(current) in seen or len(chain) >= 4:
                return StreamBackingObservation(BackingStatus.UNSUPPORTED_CHAIN,started,_NOW(),tuple(chain))
            seen.add(id(current))
            current=descriptor.__get__(current,cls)
        stage='FILENO'
        fd=_FILENO(current)
        stage='FSTAT'
        actual=_FSTAT(fd)
        facts=StreamBackingStat(actual.st_dev,actual.st_ino,actual.st_mode,actual.st_rdev,actual.st_size)
        return StreamBackingObservation(BackingStatus.OBSERVED,started,_NOW(),tuple(chain),fd,facts)
    except (OSError,ValueError) as error:
        failure=StreamBackingFailure(stage,type(error).__name__,error.errno if isinstance(error,OSError) else None)
        status={'BACKING_CHAIN':BackingStatus.CHAIN_FAILED,'FILENO':BackingStatus.FILENO_FAILED,'FSTAT':BackingStatus.FSTAT_FAILED}[stage]
        return StreamBackingObservation(status,started,_NOW(),tuple(chain),fd if stage=='FSTAT' else None,failure=failure)


# Observe exact builtin stream interfaces without wrapping application objects.
#
# This optional channel attaches resource/effect interfaces to existing CALL
# attempts. It reads no stream contents, descriptor table, cursor, or file name;
# the historical profile never calls fileno/tell/flush on behalf of the program.
# An explicit backing profile probes only saved builtin FileIO fileno plus fstat.
# C_RETURN is language
# completion only, not byte count, visibility, persistence or GPU readiness.
PROFILE = 'SCAR_BUILTIN_STREAM_INTERFACES_V1'
BACKING_PROFILE = 'SCAR_BUILTIN_STREAM_BACKING_OBSERVATIONS_V1'
SCHEMA = 'scar.stream-interface-collection'
_TYPES = {io.BytesIO: 'MEMORY_BYTES', io.StringIO: 'MEMORY_TEXT',
          io.FileIO: 'DESCRIPTOR_STREAM', io.BufferedReader: 'BUFFERED_STREAM',
          io.BufferedWriter: 'BUFFERED_STREAM', io.BufferedRandom: 'BUFFERED_STREAM',
          io.BufferedRWPair: 'BUFFERED_STREAM', io.TextIOWrapper: 'TEXT_STREAM'}
_READS = frozenset(('read', 'read1', 'readline', 'readlines', 'readinto', 'readinto1'))
_WRITES = frozenset(('write', 'writelines'))
_NAMES = _READS | _WRITES | frozenset(('seek', 'truncate', 'flush', 'close'))
_DESCRIPTORS = {}
for _type in _TYPES:
    for _name in _NAMES:
        try:
            _descriptor = inspect.getattr_static(_type, _name)
        except AttributeError:
            continue
        if type(_descriptor) in (types.MethodDescriptorType, types.WrapperDescriptorType):
            _DESCRIPTORS.setdefault(_descriptor, set()).add(_type)


@dataclass(frozen=True, slots=True)
class StreamContract:
    method: str
    reads: tuple[str, ...]
    may_write: tuple[str, ...]
    result_role: str
    labels: tuple[str, ...] = ('IO', 'STATE')
    effects_complete: bool = False
    may_raise: bool = True


def stream_contract(method):
    """Declared interface effects; unknown implementation effects stay open."""
    if method in _READS:
        if method.startswith('readinto'):
            return StreamContract(method, ('CONTENT', 'CURSOR', 'HANDLE', 'ARGUMENT_BUFFER'),
                                  ('CURSOR', 'BUFFER', 'ARGUMENT_BUFFER_CONTENT'), 'READ_COUNT')
        return StreamContract(method, ('CONTENT', 'CURSOR', 'HANDLE'), ('CURSOR', 'BUFFER'),
                              'READ_RESULT')
    if method in _WRITES:
        return StreamContract(method, ('ARGUMENT_DATA', 'HANDLE', 'CURSOR'),
                              ('CONTENT', 'CURSOR', 'BUFFER'), 'WRITE_RESULT')
    if method == 'seek':
        return StreamContract(method, ('HANDLE', 'CURSOR'), ('CURSOR', 'BUFFER'), 'CURSOR_RESULT')
    if method == 'truncate':
        return StreamContract(method, ('HANDLE', 'CURSOR'), ('CONTENT', 'BUFFER'), 'SIZE_RESULT')
    if method == 'flush':
        return StreamContract(method, ('HANDLE', 'BUFFER'), ('BUFFER', 'BACKING_CONTENT'), 'FLUSH_RESULT')
    if method == 'close':
        return StreamContract(method, ('HANDLE', 'BUFFER'), ('HANDLE', 'BUFFER', 'BACKING_CONTENT'), 'CLOSE_RESULT')
    raise ValueError('unsupported builtin stream method')


def _receiver(target, arg0, missing):
    """Recognize the actual C target and receiver; invoke no user getter."""
    if type(target) is types.MethodDescriptorType:
        receiver = None if missing else arg0
        if any(type(receiver) is cls for cls in _DESCRIPTORS.get(target, ())):
            return receiver, target.__name__
    elif type(target) is types.BuiltinFunctionType:
        receiver, name = target.__self__, target.__name__
        if any(type(receiver) is cls for cls in _TYPES) and name in _NAMES:
            descriptor = inspect.getattr_static(type(receiver), name, None)
            if (descriptor in _DESCRIPTORS
                    and target.__qualname__ == descriptor.__qualname__):
                return receiver, name
    return None


class StreamInterfaceObserver:
    """Small optional sibling channel; canonical calls keep their original IDs."""
    def __init__(self, operand, *, backing_observations=None):
        if backing_observations is None:
            selection=os.environ.get('SCAR_STREAM_BACKING_OBSERVATIONS','0')
            if selection not in ('0','1'):
                raise ValueError('SCAR_STREAM_BACKING_OBSERVATIONS must be 0 or 1')
            backing_observations=selection=='1'
        if type(backing_observations) is not bool:
            raise TypeError('stream backing observations must be an explicit bool')
        self.backing_observations=backing_observations
        from scar.trace.stream_interfaces import BackingStatus
        self.backing_counts={status.value:0 for status in BackingStatus} if backing_observations else None
        self.operand, self.recorder = operand, operand.recorder
        self.path = self.recorder.out / f'stream-interfaces.{self.recorder.pid}.jsonl'
        self.meta = self.recorder.out / f'stream-interface-meta.{self.recorder.pid}.json'
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.pending = {}
        self.buffer = bytearray()
        self.digest = sha256()
        self.counts = {'BEGIN': 0, 'C_RETURN': 0, 'C_RAISE': 0, 'UNOBSERVED': 0}
        self.records = self.bytes = 0
        self.errors = []
        self.stopped = self.closed = False

    def _flush(self):
        position = 0
        while position < len(self.buffer):
            written = os.write(self.fd, memoryview(self.buffer)[position:])
            if written <= 0:
                raise OSError('stream sidecar write made no progress')
            position += written
        self.buffer.clear()

    def _emit(self, kind, fields):
        if self.stopped or self.closed or os.getpid() != self.recorder.pid:
            return
        raw = (json.dumps({'sequence': self.records + 1, 'kind': kind, **fields},
            ensure_ascii=True, separators=(',', ':'), allow_nan=False) + '\n').encode('ascii')
        if (self.records >= self.operand.limits.max_events
                or self.bytes + len(raw) > self.operand.limits.max_bytes):
            self.stopped = True
            return
        if len(self.buffer) + len(raw) > 65536:
            self._flush()
        self.buffer.extend(raw)
        self.digest.update(raw)
        self.records += 1
        self.bytes += len(raw)
        self.counts[kind] += 1

    def begin(self, target, arg0, missing, owner, site, attempt, reference):
        if self.stopped or self.closed:
            return
        try:
            found = _receiver(target, arg0, missing)
            if found is None:
                return
            receiver, name = found
            key = threading.get_ident(), owner, site['offset']
            if key in self.pending:
                raise ValueError('unclosed stream call at the same invocation/site')
            token = reference(receiver)
            row = {'attempt_index': attempt, 'invocation': encode(InvocationKey(self.recorder.pid, key[0], owner)),
                'site': site, 'receiver_reference': token, 'receiver_kind': _TYPES[type(receiver)],
                'receiver_type': type(receiver).__module__ + '.' + type(receiver).__qualname__,
                'contract': encode(stream_contract(name)),
                'reference_identity_qualified': token.startswith('weak:'),
                'arguments_and_results_observed': False, 'backing_resource_identity_established': False}
            if self.backing_observations:
                from scar.trace.stream_interfaces import observe_stream_backing
                observation=observe_stream_backing(receiver)
                row['backing_observation']=encode(observation)
            # Retain normalized IDs and metadata only, never receiver/target/frame.
            self._emit('BEGIN', row)
            if not self.stopped:
                if self.backing_observations:
                    self.backing_counts[observation.status.value]+=1
                self.pending[key] = (id(target), row)
        except Exception as error:
            self.errors.append(type(error).__name__)
            self.stopped = True

    def complete(self, code, offset, target, frame, kind):
        if self.stopped or self.closed or os.getpid() != self.recorder.pid:
            return
        try:
            if frame is None or frame.f_code is not code:
                return
            state = self.operand._event_owner(frame, code)
            if state is None:
                return
            key = threading.get_ident(), state[0], offset
            pending = self.pending.get(key)
            if pending is None:
                return
            if pending[0] != id(target):
                raise ValueError('stream completion target differs from the observed call')
            self.pending.pop(key)
            self._emit(kind, {'attempt_index': pending[1]['attempt_index'],
                'invocation': pending[1]['invocation'], 'receiver_reference': pending[1]['receiver_reference']})
        except Exception as error:
            self.errors.append(type(error).__name__)
            self.stopped = True

    def close(self, legacy_inputs):
        if self.closed or os.getpid() != self.recorder.pid:
            return
        try:
            for _, row in self.pending.values():
                self._emit('UNOBSERVED', {'attempt_index': row['attempt_index'], 'invocation': row['invocation'],
                    'receiver_reference': row['receiver_reference']})
            self.pending.clear()
            self._flush()
            document = {'schema': SCHEMA, 'schema_version': 1, 'profile': PROFILE,
                'process_id': self.recorder.pid, 'project_root': self.recorder.root,
                'file': {'uri': str(self.path), 'sha256': self.digest.hexdigest(), 'size_bytes': self.bytes,
                         'rows': self.records}, 'counts': self.counts, 'errors': self.errors,
                'stopped': self.stopped, 'legacy_inputs': encode(legacy_inputs),
                'scope': 'exact builtin stream calls at retained CALL sites; unplanned interiors and implicit IO unexpanded',
                'tensor_or_stream_contents_read': False, 'file_descriptor_table_replayed': False,
                'source_body_or_content_proof_established': False}
            if self.backing_observations:
                document.update(schema_version=2,profile=BACKING_PROFILE,backing_counts=self.backing_counts)
            with self.meta.open('x', encoding='utf-8') as stream:
                json.dump(document, stream, ensure_ascii=True, indent=2, allow_nan=False)
                stream.write('\n')
        finally:
            os.close(self.fd)
            self.closed = True


def read_stream_interfaces(metadata, *, expected_sha256, max_bytes=64_000_000):
    """Read closed observations and join their exact original CALL attempts."""
    from scar.trace.reader import open_binary
    from collections import Counter
    path = Path(metadata).absolute()
    raw = path.read_bytes()
    if len(raw) > 2_000_000 or sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('stream metadata size/hash differs')
    doc = json.loads(raw)
    backing=doc.get('schema_version')==2 and doc.get('profile')==BACKING_PROFILE
    if (doc.get('schema') != SCHEMA or type(doc.get('schema_version')) is not int
            or not (backing or doc.get('schema_version')==1 and doc.get('profile')==PROFILE)):
        raise ValueError('unsupported stream interface metadata')
    ref = doc['file']
    if ref['uri'] != str(path.parent / f"stream-interfaces.{doc['process_id']}.jsonl"):
        raise ValueError('stream observation belongs to another namespace')
    def read_file(name, size, digest):
        if type(size) is not int or not 0 <= size <= max_bytes:
            raise ValueError('stream input byte ceiling exceeded')
        with open_binary(name) as stream:
            data = stream.read(size + 1)
        if len(data) != size or sha256(data).hexdigest() != digest:
            raise ValueError('stream input receipt differs')
        return data
    data = read_file(ref['uri'], ref['size_bytes'], ref['sha256'])
    rows = [json.loads(line) for line in data.splitlines()]
    if len(rows) != ref['rows'] or any(row['sequence'] != n for n, row in enumerate(rows, 1)):
        raise ValueError('stream event sequence differs')
    if dict(Counter(row['kind'] for row in rows)) != {k: v for k, v in doc['counts'].items() if v}:
        raise ValueError('stream event kinds/counts differ from closed metadata')
    if backing:
        from scar.trace.stream_interfaces import StreamBackingObservation,BackingStatus
        declared=doc.get('backing_counts')
        if (type(declared) is not dict or set(declared)!={status.value for status in BackingStatus}
                or any(type(n) is not int or n<0 for n in declared.values())
                or sum(declared.values())!=doc['counts']['BEGIN']):
            raise ValueError('stream backing counts differ')
        observations={}
        for row in rows:
            if row['kind']=='BEGIN':
                actual=decode(StreamBackingObservation,row['backing_observation'])
                if actual.chain_types[0]!=row['receiver_type']:
                    raise ValueError('stream backing receiver differs from original BEGIN')
                observations[row['sequence']]=actual
            elif 'backing_observation' in row:
                raise ValueError('stream backing observation is not at original BEGIN')
        counted=Counter(actual.status.value for actual in observations.values())
        if any(counted[status]!=n for status,n in declared.items()):
            raise ValueError('stream backing status/count differs from actual records')
    elif 'backing_counts' in doc or any('backing_observation' in row for row in rows):
        raise ValueError('historical stream profile cannot contain backing observations')
    calls = {row['attempt_index']: row for row in rows if row['kind'] == 'BEGIN'}
    if len(calls) != doc['counts']['BEGIN']:
        raise ValueError('duplicate stream BEGIN or count mismatch')
    original = next(r for r in doc['legacy_inputs'] if r['family'] == 'operands')
    source = read_file(original['path'], original['raw_bytes'], original['raw_sha256'])
    verified = {}
    for number, line in enumerate(source.splitlines(), 1):
        event = json.loads(line)
        index = event.get('index')
        if index not in calls:
            continue
        call = calls[index]
        invocation = decode(InvocationKey, call['invocation'])
        decode(CapturedSite, call['site'])
        if (event.get('kind') != 'CALL_ATTEMPT' or event['site'] != call['site']
                or (event['process_id'], event['thread_id'], event['invocation_id'])
                   != (invocation.process_id, invocation.thread_id, invocation.invocation_id)
                or call['contract'] != encode(stream_contract(call['contract']['method']))):
            raise ValueError('stream interface differs from original CALL identity/site or contract')
        verified[index] = {'path': original['path'], 'line_number': number,
                           'line_sha256': sha256(line + b'\n').hexdigest(), 'index': index}
    if set(verified) != set(calls):
        raise ValueError('stream interface has no original CALL attempt')
    completed = {}
    for row in rows:
        if row['kind'] == 'BEGIN':
            continue
        index = row['attempt_index']
        if index not in calls or index in completed or row['kind'] not in ('C_RETURN', 'C_RAISE', 'UNOBSERVED'):
            raise ValueError('stream completion does not select one original pending call')
        if any(row[k] != calls[index][k] for k in ('invocation', 'receiver_reference')):
            raise ValueError('stream completion receiver/invocation differs')
        completed[index] = row['kind']
    if backing:
        from scar.ir.program.providers import ProviderArtifact,ProviderFactReference
        artifact=ProviderArtifact(ref['uri'],ref['sha256'],ref['size_bytes'],'scar.stream-interfaces.events.v2')
        calls={index:{**row,'backing_source':encode(ProviderFactReference(artifact,
            '#/lines/'+str(row['sequence'])+'/backing_observation'))}for index,row in calls.items()}
    result = {'schema': 'scar.stream-interface-query', 'schema_version': 2 if backing else 1,
        'metadata': {'uri': str(path), 'sha256': expected_sha256}, 'counts': doc['counts'],
        'errors': doc['errors'], 'stopped': doc['stopped'],
        'calls': [{**call, 'attempt_source': verified[index], 'completion': completed.get(index, 'UNOBSERVED')}
                  for index, call in calls.items()],
        'qualification': 'RETAINED_CALL_RESOURCE_AND_DECLARED_STREAM_EFFECT_INTERFACE',
        'effect_implementation_or_contents_proved': False, 'whole_model_established': False}
    if backing:
        result['backing_counts']=doc['backing_counts']
    return result

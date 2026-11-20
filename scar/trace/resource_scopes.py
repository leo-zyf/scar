"""Optional, owned CALL boundary markers for exact OS/source association.

Each small binary record is one pwrite at its record offset. The OS observer
can match that offset to an original CALL without timestamp proximity or a
reconstructed shared FD table. Default capture does not enable this channel.
Only collector bytes and numeric event identities are retained.
"""
from dataclasses import dataclass
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import secrets
import struct
import threading
import types

from scar.ir.program.call_edges import InvocationKey
from scar.ir.program.resource_scopes import ScopeEventKind,ScopeEvent
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import encode
from scar.trace.files import open_binary


SCHEMA='scar.resource-scope-markers'
MAGIC=b'SCARRS01'
RECORD=struct.Struct('<8s8Q')
_WIRE_KINDS={
    ScopeEventKind.BEGIN:1,
    ScopeEventKind.C_RETURN:2,
    ScopeEventKind.C_RAISE:3,
    ScopeEventKind.PY_EXIT:4,
}
_KINDS={v:k for k,v in _WIRE_KINDS.items()}
_SUSPENDABLE = (inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR
                | inspect.CO_ITERABLE_COROUTINE)


@dataclass(frozen=True,slots=True)
class ResourceScopeMarkers:
    metadata: dict
    artifact: ProviderArtifact
    events: tuple[ScopeEvent,...]


class ResourceScopeObserver:
    C_RETURN=ScopeEventKind.C_RETURN
    C_RAISE=ScopeEventKind.C_RAISE

    def __init__(self,operand):
        self.operand=operand;self.pid=operand.pid;self.nonce=secrets.token_hex(16)
        self.path=operand.recorder.out/f'resource-scopes.{self.pid}.{self.nonce}.bin'
        self.meta=operand.recorder.out/f'resource-scope-meta.{self.pid}.json'
        self.fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        self.sequence=0;self.pending={};self.digest=sha256();self.errors=[]
        self.stopped=self.closed=False;self.completed=0
        self.python_frames = {}
        self.python_entry_candidate = None
        self.schema_version = 1

    def _emit(self,scope,kind,owner,attempt,thread):
        native=threading.get_native_id();sequence=self.sequence+1
        row=RECORD.pack(MAGIC,sequence,scope,_WIRE_KINDS[kind],thread,native,owner,attempt,self.pid)
        if os.pwrite(self.fd,row,(sequence-1)*RECORD.size)!=RECORD.size:
            raise OSError('incomplete resource scope marker write')
        self.digest.update(row);self.sequence=sequence
        return sequence

    def begin(self,target,code,offset,owner,attempt):
        if (self.closed or self.stopped or os.getpid()!=self.pid
                or type(target)is types.FunctionType
                or type(target)is types.MethodType and type(target.__func__)is types.FunctionType):return
        if not self._reserve_boundary_pair():return
        try:
            thread=threading.get_ident();scope=self.sequence+1
            self._emit(scope,ScopeEventKind.BEGIN,owner,attempt,thread)
            self.pending.setdefault((thread,id(code),offset),[]).append((scope,id(target),owner,attempt))
        except Exception as error:self.errors.append(type(error).__name__);self.stopped=True

    def _reserve_boundary_pair(self):
        # Leave unfinished scopes open on a budget stop. Matching a later
        # completion after an unrecorded recursive begin would be unsound.
        pending = self.sequence - 2 * self.completed
        if (self.sequence+pending+2)*RECORD.size>self.operand.limits.max_bytes:
            self.stopped = True
            return False
        return True

    def stage_python_entry(self, frame, owner, attempt):
        """Stage the current frame's already recorded exact Python entry."""
        if self.closed or self.stopped or frame.f_code.co_flags & _SUSPENDABLE:
            return
        self.python_entry_candidate = (threading.get_ident(), id(frame), owner, attempt)

    def commit_python_entry(self, frame):
        candidate = self.python_entry_candidate
        if candidate is None:
            return
        if self.closed or self.stopped or self.operand._stopped or self.operand._io_failed:
            return
        if os.getpid() != self.pid:
            return
        thread, identity, owner, attempt = candidate
        key = threading.get_ident(), id(frame)
        if key != (thread, identity) or key in self.python_frames:
            self.errors.append('PYTHON_ENTRY_FRAME_CHANGED')
            self.stopped = True
            return
        if not self._reserve_boundary_pair():
            return
        try:
            scope = self.sequence + 1
            self._emit(scope, ScopeEventKind.BEGIN, owner, attempt, thread)
            self.python_frames[key] = scope, owner, attempt
            self.schema_version = 2
        except Exception as error:
            self.errors.append(type(error).__name__)
            self.stopped = True

    def tracks_python(self, identity):
        return identity in self.python_frames

    def complete_python(self, identity):
        pending = self.python_frames.get(identity)
        if pending is None:
            return
        if (self.closed or self.stopped or self.operand._stopped or self.operand._io_failed
                or os.getpid() != self.pid):
            return
        scope, owner, attempt = pending
        try:
            # A profile return can be an unwind. For nonsuspendable frames
            # it is a physical exit, without a claim about result or effects.
            self._emit(scope, ScopeEventKind.PY_EXIT, owner, attempt, identity[0])
            del self.python_frames[identity]
            self.completed += 1
        except Exception as error:
            self.errors.append(type(error).__name__)
            self.stopped = True

    def complete(self,code,offset,target,kind):
        if self.closed or self.stopped or self.operand._stopped or os.getpid()!=self.pid:return
        thread=threading.get_ident();key=thread,id(code),offset;rows=self.pending.get(key)
        if not rows:return
        scope,identity,owner,attempt=rows[-1]
        if identity!=id(target):
            self.errors.append('COMPLETION_TARGET_CHANGED');self.stopped=True;return
        try:
            self._emit(scope,kind,owner,attempt,thread);rows.pop();self.completed+=1
            if not rows:del self.pending[key]
        except Exception as error:self.errors.append(type(error).__name__);self.stopped=True

    def after_fork(self):
        if os.getpid()==self.pid:return
        self.closed=self.stopped=True;self.pending.clear()
        self.python_frames.clear()
        self.python_entry_candidate = None
        try:os.close(self.fd)
        except OSError:pass

    def close(self,legacy_inputs):
        if self.closed or os.getpid()!=self.pid:return
        self.closed=True;os.close(self.fd)
        if not legacy_inputs:self.errors.append('OPERAND_INPUT_HANDOFF_UNAVAILABLE')
        document=dict(schema=SCHEMA,schema_version=self.schema_version,process_id=self.pid,nonce=self.nonce,
            file=dict(uri=str(self.path),sha256=self.digest.hexdigest(),size_bytes=self.sequence*RECORD.size),
            record_format='<8s8Q',record_bytes=RECORD.size,records=self.sequence,
            marker_fd=self.fd,completed=self.completed,
            unclosed=sum(len(v) for v in self.pending.values())+len(self.python_frames),
            stopped=self.stopped,errors=self.errors,legacy_inputs=encode(tuple(legacy_inputs)),
            transport='OWNED_SYNCHRONOUS_PWRITE_BOUNDARIES',
            ownership_condition='COLLECTOR_DESCRIPTOR_AND_MARKER_FILE_NOT_MUTATED_BY_APPLICATION',
            syscall_matching='NATIVE_TID_FD_RECORD_OFFSET_AND_EXACT_WRITE_SIZE',
            clock_matching_used=False,resource_effects_proven=False)
        raw=(json.dumps(document,sort_keys=True,separators=(',',':'))+'\n').encode()
        with self.meta.open('xb') as stream:stream.write(raw)


def read_resource_scopes(metadata,*,expected_sha256,max_bytes=128_000_000):
    """Read the exact marker stream declared by one immutable metadata file."""
    path=Path(metadata).absolute();raw=path.read_bytes()
    if sha256(raw).hexdigest()!=expected_sha256:raise ValueError('resource scope metadata changed')
    m=json.loads(raw);pid=m['process_id'];f=m['file'];logical=Path(f['uri'])
    version=m['schema_version']
    if (type(version)is not int or version not in (1,2)
            or (m['schema'],m['record_format'],m['record_bytes'])!=(SCHEMA,'<8s8Q',RECORD.size)):
        raise ValueError('unsupported resource scope record contract')
    if (path.name!=f'resource-scope-meta.{pid}.json' or logical.parent!=path.parent
            or logical.name!=f'resource-scopes.{pid}.{m["nonce"]}.bin'
            or m['errors'] or f['size_bytes']!=m['records']*RECORD.size or f['size_bytes']>max_bytes):
        raise ValueError('resource scope owner, bytes or collector status differs')
    artifact=ProviderArtifact(str(logical),f['sha256'],f['size_bytes'],SCHEMA+'.binary.v'+str(version))
    events=[];digest=sha256()
    with open_binary(logical) as source:
        for ordinal in range(m['records']):
            row=source.read(RECORD.size);digest.update(row)
            if len(row)!=RECORD.size:raise ValueError('truncated resource scope marker')
            magic,sequence,scope,kind,thread,native,owner,attempt,process=RECORD.unpack(row)
            if magic!=MAGIC or sequence!=ordinal+1 or process!=pid or not all((scope,thread,native,owner,attempt)):
                raise ValueError('resource scope marker loses its original event identity')
            if kind not in _KINDS:raise ValueError('unknown resource scope binary event kind')
            kind=_KINDS[kind]
            if version==1 and kind is ScopeEventKind.PY_EXIT:
                raise ValueError('Python frame scope completion requires resource scope version 2')
            if (kind is ScopeEventKind.BEGIN)!=(scope==sequence) or scope>sequence:
                raise ValueError('resource scope completion does not address an earlier begin')
            events.append(ScopeEvent(sequence,scope,kind,InvocationKey(pid,thread,owner),native,attempt,
                ProviderFactReference(artifact,'#/records/'+str(ordinal))))
        if source.read(1):raise ValueError('resource scope has undeclared trailing records')
    if digest.hexdigest()!=f['sha256']:raise ValueError('resource scope marker bytes changed')
    if path.read_bytes()!=raw:raise ValueError('resource scope metadata changed during read')
    return ResourceScopeMarkers(m,artifact,tuple(events))

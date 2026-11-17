"""Explicit single-owner native leases; caller effect coverage remains required."""
from copy import deepcopy
import ctypes
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import threading
import sys
from types import SimpleNamespace

from .native_loop import _small_raw, _json
from .loop_contract import _digest
from .native_stateful_repeat import _base, _json_bytes
from .native_stateful import _Table, _Scalar, _Provider, GuardFailure
from .native_loop import _Array

SCHEMA='scar.workspace-session-runtime-contract'
ADMISSION='scar.workspace-session-runtime-admission'


def implementation_sha256():
    root=Path(__file__).resolve().parents[1]
    names=('runtime/native_workspace_session.py','runtime/native_stateful_repeat.py','runtime/stateful_guard.py',
        'runtime/native_stateful.py','runtime/stateful_contract.py','runtime/native_loop.py',
        'runtime/loop_contract.py','trace/code_snapshot.py')
    return sha256(_json_bytes([(n,sha256((root/n).read_bytes()).hexdigest())for n in names])).hexdigest()


def inspect_bundle(bundle,*,expected_admission_sha256):
    directory=Path(bundle).resolve();raw=_small_raw(directory/'admission.json')
    if not _digest(expected_admission_sha256) or sha256(raw).hexdigest()!=expected_admission_sha256:
        raise GuardFailure('session admission pin differs')
    admission=_json(raw);raw=_small_raw(directory/'runtime.json');contract=_json(raw)
    if (admission.get('schema')!=ADMISSION or admission.get('schema_version')!=1
            or admission['runtime_implementation_sha256']!=implementation_sha256()
            or sha256(raw).hexdigest()!=admission['runtime_sha256']
            or contract.get('schema')!=SCHEMA or contract.get('schema_version')!=1
            or contract['native']!=admission['native'] or contract['caller_replacement_authorized'] is not False):
        raise GuardFailure('session runtime/producer identity differs')
    from .stateful_contract import validate_contract
    validate_contract(contract['step_contract'],require_selection=False)
    if contract['step_contract']['selection'] is not None or not contract['protocol']['single_owner_thread']:
        raise GuardFailure('session must retain the unselected single-owner step contract')
    from .native_loop import _file_hash_and_header
    for key in ('base','session'):
        row=contract['native'][key]
        if Path(row['library']).name!=row['library']:
            raise GuardFailure('session library must be a local member')
        path=directory/row['library'];digest,header=_file_hash_and_header(path)
        if digest!=row['sha256'] or path.stat().st_size!=row['bytes'] or header[:6]!=b'\x7fELF\x02\x01':
            raise GuardFailure('session native library bytes differ')
    return dict(directory=str(directory),contract=contract,admission=admission,library_loaded=False)


class WorkspaceSession:
    def __init__(self,directory,contract,*,trial):
        self.directory=directory;self._session_contract=contract;self.enabled=trial is True
        self.owner=threading.get_ident();self.handle=ctypes.c_void_p();self.bindings=()
        self.base=None;self.library=None;self.closed=False;self.busy=False
        self.stats=dict(creates=0,native_calls=0,resident_calls=0,invalidations=0,releases=0,binding_changes=0)
        if self.enabled:
            self.base=_base(directory,contract['step_contract'])
            native=contract['native'];self.library=ctypes.CDLL(str(directory/native['session']['library']))
            symbols=native['symbols'];getter=getattr(self.library,symbols['plan'])
            getter.argtypes=[];getter.restype=ctypes.c_char_p
            if getter()!=native['execution_plan_sha256'].encode():raise GuardFailure('session plan getter differs')
            common=[ctypes.POINTER(_Table),ctypes.POINTER(_Array),ctypes.c_uint64,
                ctypes.POINTER(_Scalar),ctypes.c_uint64,ctypes.POINTER(_Provider),ctypes.c_uint64]
            self.create=getattr(self.library,symbols['create']);self.create.argtypes=[*common,ctypes.POINTER(ctypes.c_void_p)]
            self.step=getattr(self.library,symbols['step']);self.step.argtypes=[ctypes.c_void_p,*common,
                ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_uint64)]
            self.invalidate_native=getattr(self.library,symbols['invalidate']);self.invalidate_native.argtypes=[ctypes.c_void_p]
            self.destroy=getattr(self.library,symbols['destroy']);self.destroy.argtypes=[ctypes.POINTER(ctypes.c_void_p)]
            for fn in (self.create,self.step,self.invalidate_native,self.destroy):fn.restype=ctypes.c_int32

    @property
    def contract(self):return deepcopy(self._session_contract['step_contract'])
    @property
    def report(self):return {**self._session_contract['step_contract']['native']['ids'],'session_statistics':dict(self.stats)}
    @property
    def effects(self):return deepcopy(self._session_contract['copy_effects'])

    def _owner(self):
        if threading.get_ident()!=self.owner or self.busy:raise GuardFailure('session owner/busy contract differs')

    def invalidate(self,reason='unknown_effect'):
        self._owner();self.stats['invalidations']+=1
        if self.handle.value and self.invalidate_native(self.handle):
            self.enabled=False;raise GuardFailure('cannot invalidate session')

    def release(self,reason='lease_barrier'):
        self._owner()
        if self.handle.value:
            if self.destroy(ctypes.byref(self.handle)):
                self.enabled=False;raise GuardFailure('cannot release session')
            self.stats['releases']+=1
        self.bindings=()

    def close(self):
        if not self.closed:self.release('scope_exit');self.closed=True;self.enabled=False

    def before_fallback(self,reason):
        # Fail closed even if native cleanup itself rejects: never retain a
        # usable replica fact while the original Python implementation runs.
        try:self.release('original_kernel_fallback')
        except GuardFailure:self.enabled=False

    def note_copy(self,effect,function,actual):
        """Apply a source-qualified unary copy summary before its original call."""
        try:
            self._owner()
            from .stateful_guard import _function
            import numpy as np
            _function(effect['source'],function)
            if (function.__code__.co_argcount!=1 or function.__code__.co_kwonlyargcount
                    or function.__code__.co_flags & 0x0c
                    or type(actual)is not np.ndarray or actual.dtype!=np.dtype('float64')
                    or not actual.flags.c_contiguous or list(actual.shape)!=effect['shape']):
                raise GuardFailure('copy effect actual array guard differs')
            roots=[r for r in self._session_contract['geometry']['snapshots']
                if r['mode']!='KEEP_FULL_SNAPSHOT']
            specs=[r for r in self.contract['inputs']if r['kind']=='array']
            for index,held in enumerate(self.bindings):
                if actual is held:
                    root=next((r for r in roots if r['input_id']==specs[index]['id']),None)
                    if root is None:continue
                    (a,b),(c,d)=root['committed_roi']
                    if any(a<r[0][1] and r[0][0]<b and c<r[1][1] and r[1][0]<d for r in effect['writes']):
                        self.invalidate('overlapping_copy_write');return False
                elif np.may_share_memory(actual,held):
                    self.invalidate('different_actual_may_alias');return False
            return True
        except (ValueError,TypeError,KeyError,GuardFailure):
            self.before_fallback('unqualified_copy_effect');return False

    def run(self,arrays,scalars):
        self._owner()
        if not self.enabled or self.closed:raise GuardFailure('session is not explicitly enabled')
        monitor=getattr(sys,'monitoring',None)
        if (sys.gettrace()is not None or sys.getprofile()is not None
                or monitor is not None and any(monitor.get_tool(i)is not None for i in range(6))):
            self.before_fallback('source_observer_active')
            raise GuardFailure('active source observers are outside the session trial contract')
        ids=[r['id']for r in self.contract['inputs']if r['kind']=='array']
        try:
            current=tuple(arrays[k]for k in ids)
            if self.bindings and any(a is not b for a,b in zip(current,self.bindings)):
                self.invalidate('array_object_changed');self.stats['binding_changes']+=1
            self.busy=True
            def invoke(*args):
                if not self.handle.value:
                    status=self.create(self.base._table,*args,ctypes.byref(self.handle))
                    if status:return status
                    self.stats['creates']+=1
                completed=ctypes.c_uint64();used=ctypes.c_uint64()
                status=self.step(self.handle,self.base._table,*args,ctypes.byref(completed),ctypes.byref(used))
                if not status and completed.value!=1:raise GuardFailure('successful session call did not commit exactly one step')
                if not status:
                    self.stats['native_calls']+=1;self.stats['resident_calls']+=used.value
                return status
            facade=replace(self.base,_table=SimpleNamespace(contents=SimpleNamespace(run=invoke)))
            result=facade.run(arrays,scalars);self.bindings=current;return result
        except BaseException:
            self.busy=False;self.before_fallback('step_failed');raise
        finally:self.busy=False

    def wrap(self,original,*,owner=None,exclusive_inputs=False):
        if not self.enabled:return original
        from .stateful_guard import guarded_stateful_function
        return guarded_stateful_function(original,self,owner=owner,exclusive_inputs=exclusive_inputs,
            trial=True,on_fallback=self.before_fallback)

    def __enter__(self):return self
    def __exit__(self,*unused):self.close()


def load(bundle,*,expected_admission_sha256,trial=False):
    admitted=inspect_bundle(bundle,expected_admission_sha256=expected_admission_sha256)
    return WorkspaceSession(Path(bundle).resolve(),admitted['contract'],trial=trial)

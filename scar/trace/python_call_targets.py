"""Python call targets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import os
import sys
import threading
import types
import weakref
import dis
from scar.ir.program.python_call_targets import (
    PythonTargetKind as Kind,
    PythonCodeScope as Scope,
    ReceiverRole,
    ReceiverCondition,
    BoundReceiver,
    PythonCallTarget,
    PythonEntryBinding,
    PythonTargetGap,
    target_record,
)
from scar.ir.program.call_edges import InvocationKey
from scar.ir.record_codec import encode
from scar.trace.code_snapshot import code_key as captured_code_key, call_instruction, site
from types import CodeType


# Opt-in v2 targets for observed Python calls; no user getter/target execution.
# Use the builtin descriptor directly: isinstance(value, type) can consult a
# non-type object's user-defined __class__, and reading metaclass attributes
# normally can execute custom __getattribute__. No application getter, equality
# or repr participates in this classification.
_BUILTIN_TYPE = type
_ACTUAL_TYPE_MRO = type.__dict__['__mro__'].__get__


def _is_type_object(value):
    for base in _ACTUAL_TYPE_MRO(_BUILTIN_TYPE(value)):
        if base is _BUILTIN_TYPE:
            return True
    return False


def _invocation_record(invocation):
    return {'process_id': invocation.process_id, 'thread_id': invocation.thread_id,
            'invocation_id': invocation.invocation_id}


def _target_record(kind, value):
    """Exact V2 projection, retaining target_record's full typed validation.

    Records/layouts are frozen, but mutable wire containers are always fresh.
    This avoids generic dataclass/enum discovery on every CALL. Runtime fields
    (including receiver/default-dependent category) still come from this event.
    Unknown kind/type takes the ordinary public error path.
    """
    expected = {'TARGET': PythonCallTarget, 'ENTRY': PythonEntryBinding,
                'GAP': PythonTargetGap}.get(kind)
    if expected is None or type(value) is not expected:
        return target_record(kind, value)
    value.__post_init__()
    if kind == 'TARGET':
        from scar.trace.call_operands import _layout_record
        receiver = value.receiver
        receiver_record = None if receiver is None else {
            'role': receiver.role.value, 'reference': receiver.reference,
            'explicit_positional': receiver.explicit_positional,
            'conditions': [condition.value for condition in receiver.conditions]}
        fields = {'attempt_index': value.attempt_index,
            'caller': _invocation_record(value.caller), 'kind': value.kind.value,
            'callable_reference': value.callable_reference,
            'function_reference': value.function_reference, 'code': value.code,
            'layout': None if value.layout is None else _layout_record(value.layout),
            'receiver': receiver_record, 'scope': value.scope.value}
    elif kind == 'ENTRY':
        fields = {'attempt_index': value.attempt_index,
            'caller': _invocation_record(value.caller),
            'callee': _invocation_record(value.callee), 'entry_index': value.entry_index,
            'code': value.code, 'receiver_parameter_index': value.receiver_parameter_index,
            'receiver_reference': value.receiver_reference}
    else:
        fields = {'attempt_index': value.attempt_index, 'reason': value.reason}
    return {'schema': 'scar.python-call-targets', 'schema_version': 2,
            'kind': kind, 'value': fields}


@dataclass(frozen=True,slots=True)
class PythonTargetBudget:
    max_records:int=100_000
    max_bytes:int=32_000_000
    max_selected_codes:int=4096
    def __post_init__(self):
        if not all(type(x) is int and x>0 for x in (self.max_records,self.max_bytes,self.max_selected_codes)):
            raise ValueError('positive independent Python target budgets required')

class PythonTargetObserver:
    def __init__(self,operand,budget=PythonTargetBudget()):
        if type(budget) is not PythonTargetBudget:raise TypeError('typed PythonTargetBudget required')
        if sys.implementation.name!='cpython' or sys.version_info[:3]!=(3,14,7):raise RuntimeError('PYTHON_CALL_TARGETS_V2 requires the fixed CPython 3.14.7 CALL contract')
        budget.__post_init__();self.operand=operand;self.recorder=operand.recorder;self.budget=budget
        self.path=self.recorder.out/f'python-call-targets.{self.recorder.pid}.jsonl'
        self.meta=self.recorder.out/f'python-call-target-meta.{self.recorder.pid}.json'
        self.fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        self._deferred=None
        from scar.trace.deferred_jsonl import DeferredJsonl,snapshot_transport_enabled
        if snapshot_transport_enabled():
            self._deferred=DeferredJsonl(self.path,max_bytes=budget.max_bytes)
        self.pending={};self.codes={};self.records=self.bytes=0;self.closed=self.stopped=False
        self.counts={'TARGET':0,'ENTRY':0,'GAP':0};self.digest=hashlib.sha256()
        self._buffer=bytearray();self._buffer_limit=65536
        self._io_failed=False
        self.legacy_inputs=None
        self.native_callables = None
        probe_path = os.environ.get('SCAR_NATIVE_CALLABLE_PROBE')
        if probe_path:
            try:
                from scar.trace.native_callables import NativeCallableObserver
                self.native_callables = NativeCallableObserver(self, probe_path)
            except BaseException:
                if self._deferred is not None:self._deferred.close_capture()
                os.close(self.fd)
                self.closed = True
                raise
    def emit(self,kind,value):
        """Compatibility path for callers supplying ordinary typed records."""
        if self.closed or self.stopped:return
        self._emit_fields(kind,_target_record(kind,value)['value'])
    def _emit_fields(self,kind,fields):
        """Freeze raw collector facts; the ordinary reader validates the IR.

        Internal callbacks do not build execution/program record objects.
        Schema, field order, counters and both transports remain unchanged.
        Current code/default/receiver observations still happen at each CALL.
        """
        if self.closed or self.stopped:return False
        document={'schema':'scar.python-call-targets','schema_version':2,'kind':kind,'value':fields}
        if self._deferred is not None:
            if self.records>=self.budget.max_records or not self._deferred.append(document):
                self.stopped=True;return False
            self.bytes=self._deferred.accepted_bytes
            self.records+=1;self.counts[kind]+=1
            return True
        raw=(json.dumps(document,ensure_ascii=True,separators=(',',':'),allow_nan=False)+'\n').encode('ascii')
        if self.records>=self.budget.max_records or self.bytes+len(raw)>self.budget.max_bytes:
            self.stopped=True;return False
        try:
            if len(self._buffer)+len(raw)>self._buffer_limit:self._flush()
            if len(raw)>=self._buffer_limit:self._write_all(raw)
            else:self._buffer.extend(raw)
        except BaseException:
            self.stopped=self._io_failed=True
            raise
        self.digest.update(raw);self.bytes+=len(raw);self.records+=1;self.counts[kind]+=1
        return True
    def _write_all(self,raw):
        if os.getpid()!=self.recorder.pid:raise OSError('forked process cannot flush parent Python target records')
        offset=0
        while offset<len(raw):
            n=os.write(self.fd,raw[offset:])
            if n<=0:raise OSError('Python target sidecar made no write progress')
            offset+=n
    def _flush(self):
        if self._deferred is not None:
            result=self._deferred.finalize()
            self.bytes=result['output']['raw_bytes']
            return
        if self._buffer:self._write_all(self._buffer);self._buffer.clear()
    def gap(self,attempt,reason):self._emit_fields('GAP',{'attempt_index':attempt,'reason':reason})
    def observed_code(self,code):
        ref=self.codes.get(id(code));return ref is not None and ref() is code
    def select_code(self,code,attempt):
        if self.observed_code(code):return True
        self.codes={k:v for k,v in self.codes.items() if v() is not None}
        if len(self.codes)>=self.budget.max_selected_codes:
            self.gap(attempt,'OBSERVED_PYTHON_CODE_SCOPE_BUDGET');return False
        self.codes[id(code)]=weakref.ref(code);return True
    def on_call(self,callable_object,arg0,arg0_missing,caller_code,owner,call_site,attempt,reference,layout):
        key=threading.get_ident(),owner
        old=self.pending.pop(key,None)
        if old is not None:self.gap(old['attempt'],'PYTHON_TARGET_DID_NOT_ENTER_BEFORE_NEXT_CALL')
        func=receiver=None;kind=Kind.OTHER_UNCLASSIFIED
        if type(callable_object) is types.FunctionType:func=callable_object;kind=Kind.FUNCTION
        elif type(callable_object) is types.MethodType:
            candidate=callable_object.__func__
            if type(candidate) is types.FunctionType:func=candidate;receiver=callable_object.__self__;kind=Kind.BOUND_METHOD
        elif type(callable_object) is types.BuiltinFunctionType:kind=Kind.NATIVE_BUILTIN
        elif type(callable_object) is types.MethodWrapperType:kind=Kind.NATIVE_WRAPPER
        elif _is_type_object(callable_object):kind=Kind.TYPE_OBJECT
        call_ref=reference(callable_object)
        caller={'process_id':self.recorder.pid,'thread_id':key[0],'invocation_id':owner}
        if self.native_callables is not None:
            self.native_callables.on_call(callable_object, caller, attempt, call_ref, reference)
        if func is None:
            self._emit_fields('TARGET',{'attempt_index':attempt,'caller':caller,'kind':kind.value,
                'callable_reference':call_ref,'function_reference':None,'code':None,'layout':None,
                'receiver':None,'scope':Scope.NO_PYTHON_CODE.value});return
        code=func.__code__;code_key=captured_code_key(code);function_ref=reference(func)
        # CALL_KW observes the callable but arg0 alone does not distinguish
        # its first explicit value from an implicit method receiver. Preserve
        # target/entry identity without extending the positional CALL ABI.
        keyword_roles_open=kind is Kind.FUNCTION and call_site['opname']=='CALL_KW'
        if kind is Kind.FUNCTION and not arg0_missing:
            instruction=call_instruction(caller_code,call_site['offset'])
            unsupported=inspect.CO_VARARGS|inspect.CO_VARKEYWORDS|inspect.CO_GENERATOR|inspect.CO_COROUTINE|inspect.CO_ASYNC_GENERATOR
            # CPython3.14 expands PyMethod before _MONITOR_CALL. CALL n
            # excludes the implicit first parameter. Under this exact finite
            # no-default/nonvariadic ABI a successful matching entry needs n+1.
            if (instruction is not None and instruction[0]=='CALL'
                    and type(instruction[1]) is int and code.co_argcount==instruction[1]+1
                    and code.co_kwonlyargcount==0 and not code.co_flags & unsupported
                    and func.__defaults__ is None):
                kind=Kind.NORMALIZED_RECEIVER_FUNCTION;receiver=arg0
        receiver_ref=None if receiver is None else reference(receiver)
        scope=Scope.PROJECT if self.operand._code_inside(code) else Scope.OBSERVED_EXTERNAL_PYTHON if os.path.isabs(code.co_filename) else Scope.NONFILESYSTEM_PYTHON
        normalized=kind is Kind.NORMALIZED_RECEIVER_FUNCTION
        role=None if receiver is None else {
            'role':(ReceiverRole.IMPLICIT_CALL_RECEIVER if normalized else ReceiverRole.BOUND_METHOD_RECEIVER).value,
            'reference':receiver_ref,'explicit_positional':instruction[1] if normalized else None,
            'conditions':[ReceiverCondition.NO_INTERVENING_CALL_LAYOUT_MUTATION.value] if normalized else []}
        from scar.trace.call_operands import _layout_record
        self._emit_fields('TARGET',{'attempt_index':attempt,'caller':caller,'kind':kind.value,
            'callable_reference':call_ref,'function_reference':function_ref,'code':code_key,
            'layout':_layout_record(layout(code)),'receiver':role,'scope':scope.value})
        if keyword_roles_open:self.gap(attempt,'CALL_KW_ARGUMENT_ROLES_NOT_INFERRED')
        if not self.select_code(code,attempt):return
        try:receiver_weak=None if receiver is None else weakref.ref(receiver)
        except TypeError:receiver_weak=None
        try:method_weak=weakref.ref(callable_object) if kind is Kind.BOUND_METHOD else None
        except TypeError:method_weak=None
        self.pending[key]={'attempt':attempt,'caller':(self.recorder.pid,key[0],owner),'function':weakref.ref(func),'code':weakref.ref(code),'code_key':code_key,'site':call_site,'receiver':receiver_weak,'method':method_weak,'receiver_ref':receiver_ref,'kind':kind,'defaults_none':func.__defaults__ is None}
    def pending_for(self,frame):
        parent=frame.f_back
        state=self.operand._base_state(parent) if parent is not None else None
        if state is None:return None
        pending=self.pending.get((threading.get_ident(),state[0]))
        if pending is None:return None
        func,code=pending['function'](),pending['code']()
        if func is None or code is None or frame.f_code is not code or func.__code__ is not code:return None
        try:matched=site(parent.f_code,parent.f_lasti)==pending['site']
        except (TypeError,ValueError):matched=False
        return pending if matched else None
    def accepts_frame(self,frame):return self.pending_for(frame) is not None
    def on_frame_enter(self,frame):
        pending=self.pending_for(frame)
        if pending is None:
            if self.operand.lean_plan is not None and frame.f_back is not None:
                parent=self.operand._base_state(frame.f_back)
                lost=self.pending.pop((threading.get_ident(),parent[0]),None) if parent is not None else None
                if lost is not None:self.gap(lost['attempt'],'PYTHON_TARGET_NON_IMMEDIATE_ENTRY')
            return False
        state=self.operand._base_state(frame)
        if state is None:
            self.gap(pending['attempt'],'PYTHON_ENTRY_NOT_RETAINED_BY_FRAME_RECORDER');return False
        key=threading.get_ident(),pending['caller'][2]
        self.pending.pop(key,None)
        receiver_index=receiver_ref=None
        if pending['kind'] in (Kind.BOUND_METHOD,Kind.NORMALIZED_RECEIVER_FUNCTION):
            receiver=pending['receiver']() if pending['receiver'] is not None else None
            method=pending['method']() if pending['method'] is not None else None
            if receiver is None and type(method) is types.MethodType:receiver=method.__self__
            code=frame.f_code
            if pending['kind'] is Kind.NORMALIZED_RECEIVER_FUNCTION and pending['function']().__defaults__ is not None:
                self.gap(pending['attempt'],'CALL_RECEIVER_DEFAULT_STATE_CHANGED');return True
            if receiver is None:
                self.gap(pending['attempt'],'BOUND_RECEIVER_CONTINUITY_UNAVAILABLE');return True
            if code.co_argcount<1 or not code.co_varnames or frame.f_locals.get(code.co_varnames[0]) is not receiver:
                self.gap(pending['attempt'],'BOUND_RECEIVER_FIRST_FORMAL_MISMATCH');return True
            receiver_index=0;receiver_ref=pending['receiver_ref']
        # Project exact-function entry authority remains the original v1
        # ENTRY_HANDSHAKE. v2 adds entries only for methods/external code.
        if pending['kind'] is Kind.FUNCTION and self.operand._code_inside(frame.f_code):return True
        pid,tid,owner=pending['caller']
        observed = self._emit_fields('ENTRY',{'attempt_index':pending['attempt'],
            'caller':{'process_id':pid,'thread_id':tid,'invocation_id':owner},
            'callee':{'process_id':self.recorder.pid,'thread_id':threading.get_ident(),'invocation_id':state[0]},
            'entry_index':self.recorder.index,'code':pending['code_key'],
            'receiver_parameter_index':receiver_index,'receiver_reference':receiver_ref})
        scopes = self.operand.resource_scopes
        if observed and scopes is not None:
            scopes.stage_python_entry(frame, owner, pending['attempt'])
        return True
    def close(self):
        if self.closed or os.getpid()!=self.recorder.pid:return
        try:
            if self._io_failed:raise OSError('Python target sidecar has an earlier write failure')
            for pending in tuple(self.pending.values()):self.gap(pending['attempt'],'PYTHON_TARGET_NO_IMMEDIATE_ENTRY_RETAINED')
            self._flush()
        except BaseException:
            self.stopped=self._io_failed=True
            raise
        finally:
            if self._deferred is not None:self._deferred.close_capture()
            self.pending.clear();self._buffer.clear();os.close(self.fd);self.closed=True
        digest=(self.digest.hexdigest() if self._deferred is None
                else self._deferred.result['output']['sha256'])
        document={'schema':'scar.python-call-target-collection','schema_version':2,'capabilities':['PYTHON_CALL_TARGETS_V2','OBSERVED_PYTHON_CALLEE_SCOPE_V1'],'process_id':self.recorder.pid,'project_root':self.recorder.root,'scope_policy':'OBSERVED_EXACT_PYTHON_CALLEES_V1','budget':encode(self.budget),'counts':self.counts,'stopped':self.stopped,'file':{'path':str(self.path),'bytes':self.bytes,'sha256':digest,'rows':self.records},'legacy_inputs':encode(self.legacy_inputs),'receiver_is_separate_from_monitoring_arg0':True,'contents_state_ready_or_complete_program_established':False}
        with self.meta.open('x') as stream:json.dump(document,stream,indent=2);stream.write('\n')


# Small, versioned decoder for the load occurrences B1 can observe.
#
# This is intentionally narrower than a Python bytecode interpreter.  It only
# names ordinary local loads, the measured 3.14 dual-borrow local load, and
# global-name loads.  In particular, it does not infer stack effects for mixed
# load/store instructions, attributes, closures, or packed call forms.
_FAST_LOADS = frozenset({"LOAD_FAST", "LOAD_FAST_BORROW"})
_DUAL_BORROW = "LOAD_FAST_BORROW_LOAD_FAST_BORROW"
_GLOBAL_LOADS = frozenset({"LOAD_GLOBAL"})
CALL_FAMILY_PREFIX = "CALL"


def load_slots(code, instruction):
    """Return ordered supported load slots for one exact disassembly record.

    The result is ``(suboperand, namespace, name, slot_index)`` tuples.
    ``namespace`` is ``FAST_LOCAL`` or ``GLOBAL``; a runtime global lookup can
    resolve to either the globals or builtins dictionary.  ``()`` means this
    instruction is not a load.  ``None`` means it is a load shape this rule
    does not decode.

    CPython 3.14.7 encodes ``LOAD_FAST_BORROW_LOAD_FAST_BORROW`` as the first
    local index in the high nibble and the second index in the low nibble.
    Both indexes are checked against ``argval`` and ``co_varnames`` so a
    changed or unfamiliar encoding fails closed.
    """
    if type(code) is not CodeType or type(instruction) is not dis.Instruction:
        raise TypeError("load_slots requires exact CodeType and dis.Instruction values")

    opname = instruction.opname
    if opname in _FAST_LOADS:
        index = instruction.arg
        name = instruction.argval
        if (type(index) is not int or index < 0 or index >= code.co_nlocals
                or type(name) is not str or index >= len(code.co_varnames)
                or code.co_varnames[index] != name):
            return None
        return ((0, "FAST_LOCAL", name, index),)

    if opname == _DUAL_BORROW:
        packed = instruction.arg
        names = instruction.argval
        if type(packed) is not int or not 0 <= packed <= 0xFF:
            return None
        if type(names) is not tuple or len(names) != 2 or any(type(name) is not str for name in names):
            return None
        indexes = (packed >> 4, packed & 0xF)
        result = []
        for ordinal, (index, name) in enumerate(zip(indexes, names)):
            if (index >= code.co_nlocals or index >= len(code.co_varnames)
                    or code.co_varnames[index] != name):
                return None
            result.append((ordinal, "FAST_LOCAL", name, index))
        return tuple(result)

    if opname in _GLOBAL_LOADS:
        name = instruction.argval
        if type(name) is not str or not name:
            return None
        return ((0, "GLOBAL", name, None),)

    if opname.startswith("LOAD_"):
        return None
    return ()


def call_family_offsets(code, instructions=None):
    """Return the logical offsets of all CPython ``CALL*`` instructions.

    This set is a sparse capture selection only.  It does not mean that each
    opcode dispatches a Python callable; for example ``CALL_INTRINSIC_*`` is
    included as a bytecode location but is not an operand-route candidate.
    """
    code, instructions = _checked_instructions(code, instructions)
    return frozenset(item.offset for item in instructions
                     if item.opname.startswith(CALL_FAMILY_PREFIX))


def call_window_selection(code, instructions=None):
    """Select exact straight-line bytecode windows for fixed positional calls.

    A returned mapping is keyed by a plain ``CALL`` offset and contains the
    complete logical instruction offsets from the first required name load
    through that ``CALL``.  Every intervening instruction must be a supported
    load, ``PUSH_NULL``, or ``NOP``.  A control-flow or exception-handler edge
    may enter at the first instruction, but not inside the window.

    ``expiry_offsets`` contains the normal successor of every ``CALL*``
    instruction and every exception-table handler target.  These locations
    must remain monitored so a pending call attempt can be expired on a
    normal continuation or exception path, even when no attempt was pending
    the first time the location ran.

    This is a bytecode candidate filter, not a source or runtime proof.  The
    source matcher must independently replay the code and match each window
    to the exact source call and ordered name occurrences.
    """
    code, instructions = _checked_instructions(code, instructions)
    by_offset = {item.offset: index for index, item in enumerate(instructions)}
    jump_targets = {item.offset for item in instructions if item.is_jump_target}
    call_offsets = {
        index for index, item in enumerate(instructions)
        if item.opname.startswith(CALL_FAMILY_PREFIX)
    }
    expiry_offsets = set()
    for index in call_offsets:
        if index + 1 < len(instructions):
            expiry_offsets.add(instructions[index + 1].offset)

    try:
        exception_entries = dis.Bytecode(code).exception_entries
    except BaseException as error:
        raise ValueError("cannot decode CPython exception table for call windows") from error
    exception_targets = set()
    for entry in exception_entries:
        target = entry.target
        if type(target) is not int or target < 0:
            raise ValueError("exception table has a noncanonical handler target")
        if target not in by_offset:
            raise ValueError("exception table handler is not a logical instruction offset")
        exception_targets.add(target)
        expiry_offsets.add(target)

    windows = {}
    for position, call in enumerate(instructions):
        if call.opname != "CALL" or type(call.arg) is not int or call.arg < 0:
            continue
        required_loads = call.arg + 1
        count = 0
        reversed_window = []
        valid = True
        # Walk the shared instruction tuple by index.  A slice per CALL makes
        # a many-call code object allocate/copy every preceding prefix.
        for previous_position in range(position - 1, -1, -1):
            instruction = instructions[previous_position]
            slots = load_slots(code, instruction)
            if slots is None:
                valid = False
                break
            if slots:
                count += len(slots)
            elif instruction.opname not in ("PUSH_NULL", "NOP"):
                valid = False
                break
            reversed_window.append(instruction)
            if count >= required_loads:
                break
        if not valid or count != required_loads:
            continue
        window = tuple(item.offset for item in reversed(reversed_window)) + (call.offset,)
        if len(set(window)) != len(window) or tuple(sorted(window)) != window:
            continue
        interior = set(window[1:])
        if interior & (jump_targets | exception_targets):
            continue
        windows[call.offset] = window

    # Keep this sanity check close to disassembly decoding: callback sites
    # must always correspond to an instruction in the same exact code object.
    if any(offset not in by_offset for offsets in windows.values() for offset in offsets):
        raise ValueError("call-window plan contains an unknown instruction offset")
    return windows, frozenset(expiry_offsets)


def _checked_instructions(code, instructions):
    if type(code) is not CodeType:
        raise TypeError("call-window selection requires an exact CodeType")
    if instructions is None:
        instructions = tuple(dis.get_instructions(code))
    elif type(instructions) in (tuple, list):
        instructions = tuple(instructions)
    else:
        raise TypeError("instructions must be a list/tuple of exact dis.Instruction values")
    previous = -1
    for item in instructions:
        if type(item) is not dis.Instruction:
            raise TypeError("instructions must contain exact dis.Instruction values")
        if type(item.offset) is not int or item.offset <= previous:
            raise ValueError("disassembly instruction offsets must be strictly increasing")
        if type(item.opname) is not str:
            raise ValueError("disassembly opcode name must be exact text")
        previous = item.offset
    return code, instructions


__all__ = [
    'CALL_FAMILY_PREFIX',
    'call_family_offsets',
    'call_window_selection',
    'load_slots',
    'PythonTargetBudget',
    'PythonTargetObserver',
]

"""Bounded, owner-thread RecordFunction scopes for CPython native calls."""

import os
import threading
import types


PROFILE = 'scar.torch-call-scope-v1'
LIMIT = 1024
_NATIVE_CALL_TYPES = (types.BuiltinFunctionType, types.MethodDescriptorType,
                      types.WrapperDescriptorType, types.MethodWrapperType)


class TorchCallScopes:
    def __init__(self, torch, *, nonce, pid):
        from scar.trace.dispatch_evidence import PROVIDER_REVISION

        if torch.version.git_version != PROVIDER_REVISION:
            raise ValueError('unsupported PyTorch RecordFunction provider')
        self.enter = torch.autograd._record_function_with_args_enter
        self.exit = torch.autograd._record_function_with_args_exit
        self.owner = threading.get_ident()
        self.native_tid = threading.get_native_id()
        self.pid = pid
        self.prefix = f'scar-native-call:{nonce}:{pid}:'
        self.pending = []
        self.rows = []
        self.errors = []
        self.budget_exhausted = False
        self.closed = False

    def begin(self, frame, target, invocation, site, attempt):
        if (self.closed or self.errors or self.budget_exhausted or type(target) not in _NATIVE_CALL_TYPES
                or threading.get_ident() != self.owner or os.getpid() != self.pid):
            return
        if len(self.rows) >= LIMIT:
            self.budget_exhausted = True
            return
        try:
            handle = self.enter(self.prefix + str(attempt))
        except Exception:
            self.errors.append('MARKER_ENTER_FAILED')
            return
        row = dict(attempt=attempt, invocation=invocation, site=site, outcome='UNCLOSED')
        self.rows.append(row)
        self.pending.append((id(frame), id(frame.f_code), site['offset'], id(target), handle, row))

    def complete(self, frame, code, offset, target, outcome):
        if (not self.pending or self.closed or threading.get_ident() != self.owner
                or os.getpid() != self.pid):
            return
        frame_id, code_id, begin_offset, target_id, handle, row = self.pending[-1]
        if id(frame) != frame_id:
            return
        if (id(code), offset, id(target)) != (code_id, begin_offset, target_id):
            self.errors.append('MARKER_COMPLETION_IDENTITY_CHANGED')
            return
        self.pending.pop()
        try:
            self.exit(handle)
        except Exception:
            self.errors.append('MARKER_EXIT_FAILED')
            return
        row['outcome'] = outcome

    def close(self):
        self.closed = True
        while self.pending:
            # Release the owned provider scope before profiler shutdown. The
            # cleanup has no original CPython completion callback.
            *_, handle, _row = self.pending.pop()
            try:
                self.exit(handle)
            except Exception:
                self.errors.append('UNCLOSED_MARKER_EXIT_FAILED')
        return dict(profile=PROFILE, owner_python_ident=self.owner, native_tid=self.native_tid,
                    rows=self.rows, limit=LIMIT, budget_exhausted=self.budget_exhausted,
                    errors=sorted(set(self.errors)))

"""Bounded CALL metadata on explicitly supplied original Python code objects.

No global monitoring, exec audit hook, argument values or callee-body tracing.
Events are call attempts, never proof of completion, effects or all writes.
The caller owns source/F4 qualification and any lease invalidation policy.
"""
import os
import sys
import threading
from types import CodeType

from scar.trace.call_binding import target_metadata, _type_metadata
from scar.trace.code_snapshot import describe_code, site

_OWNERS = {}
_GETTRACE, _GETPROFILE = sys.gettrace, sys.getprofile


class LocalCallRoutes:
    def __init__(self, codes, *, max_events=4096, on_before_call=None, on_invalidate=None):
        if (type(codes) not in (tuple, list) or not 1 <= len(codes) <= 64
                or any(type(code) is not CodeType for code in codes)
                or len({id(code) for code in codes}) != len(codes)):
            raise ValueError('one to sixty-four distinct verified CodeType objects required')
        if type(max_events) is not int or not 1 <= max_events <= 1_000_000:
            raise ValueError('positive bounded CALL metadata budget required')
        if any(hook is not None and not callable(hook) for hook in (on_before_call, on_invalidate)):
            raise TypeError('optional callback/invalidation hooks must be callable')
        self.codes = tuple(codes); self._codes = {id(code): code for code in codes}
        self._descriptions = [describe_code(code) for code in codes]
        self.max_events = max_events
        self.on_before_call, self.on_invalidate = on_before_call, on_invalidate
        self.pid = os.getpid(); self.tool_id = None
        self.tool_name = 'scar.local-call-routes.%s.%s' % (self.pid, id(self))
        self.monitor = getattr(sys, 'monitoring', None)
        self.records = []; self.errors = []; self.status = 'NEW'; self.stop_reason = None
        self.closed = False; self.permanent_disable = False; self.active = False; self._busy = False
        self.callback_identity_verified_on_close = False
        self._callback_registered = False
        self._callback_ref = self._on_call

    def _error(self, stage, error):
        if len(self.errors) < 16:
            self.errors.append(dict(stage=stage, error=error if type(error) is str else type(error).__name__))

    def _owns_slot(self):
        return (self.tool_id is not None and _OWNERS.get(self.tool_id) is self
                and self.monitor.get_tool(self.tool_id) == self.tool_name)

    def owns_events(self):
        """Check object/name/masks; the public API has no callback read getter."""
        try:
            return (self._owns_slot() and self.monitor.get_events(self.tool_id) == 0
                    and all(self.monitor.get_local_events(self.tool_id, code) == self.monitor.events.CALL
                            for code in self.codes))
        except BaseException:
            return False

    def _foreign_observer(self):
        if _GETTRACE() is not None or _GETPROFILE() is not None:
            return True
        for tool in range(6):
            if tool == self.tool_id:
                continue
            if (self.monitor.get_tool(tool) is not None or self.monitor.get_events(tool)
                    or any(self.monitor.get_local_events(tool, code) for code in self.codes)):
                return True
        return False

    def _disable_owned_events(self):
        try:
            if self._owns_slot():
                for code in self.codes:
                    mask = self.monitor.get_local_events(self.tool_id, code)
                    if mask == self.monitor.events.CALL:
                        self.monitor.set_local_events(self.tool_id, code, 0)
                    elif mask:
                        self._error('disable', 'OWNED_LOCAL_MASK_CHANGED_NOT_CLEARED')
        except BaseException as error:
            self._error('disable', error)

    def _invalidate(self, reason):
        if self.permanent_disable:
            return
        self.permanent_disable = True; self.active = False; self.stop_reason = reason
        self.status = ('BUDGET_EXHAUSTED' if reason == 'budget' else 'CLOSED' if reason == 'close'
                       else 'INVALIDATED')
        # Notify before monitoring stops: no upper lease may remain valid once
        # future CALL events are no longer checked. Notification errors are
        # recorded; they never propagate into the observed program.
        try:
            if self.on_invalidate is not None:
                self.on_invalidate(reason)
        except BaseException as error:
            self._error('on_invalidate', error)
        finally:
            self._disable_owned_events()

    def start(self):
        if self.active or self.closed or self.permanent_disable:
            return self
        try:
            if (self.monitor is None or sys.implementation.name != 'cpython'
                    or sys.version_info[:2] != (3, 14) or not hasattr(self.monitor, 'MISSING')):
                self._invalidate('unsupported_interpreter'); return self
            if self._foreign_observer():
                self._invalidate('foreign_observer'); return self
            for tool in range(5, -1, -1):
                if self.monitor.get_tool(tool) is not None:
                    continue
                try:
                    self.monitor.use_tool_id(tool, self.tool_name)
                except ValueError:
                    continue
                self.tool_id = tool; _OWNERS[tool] = self; break
            if self.tool_id is None:
                self._invalidate('no_free_tool'); return self
            if self.monitor.get_events(self.tool_id) != 0:
                self._invalidate('global_mask_changed'); return self
            self.monitor.register_callback(self.tool_id, self.monitor.events.CALL, self._callback_ref)
            self._callback_registered = True
            for code in self.codes:
                if self.monitor.get_local_events(self.tool_id, code) != 0:
                    self._invalidate('local_mask_changed'); return self
                self.monitor.set_local_events(self.tool_id, code, self.monitor.events.CALL)
            self.active = True; self.status = 'ACTIVE'
        except BaseException as error:
            self._error('start', error); self._invalidate('start_error')
        return self

    def _on_call(self, code, offset, callable_object, arg0):
        try:
            if not self.active or self.permanent_disable:
                return
            if self._busy:
                self._invalidate('reentrant_callback'); return
            self._busy = True
            if os.getpid() != self.pid:
                self._invalidate('process_changed'); return
            if not self.owns_events() or self._foreign_observer():
                self._invalidate('observer_or_ownership_changed'); return
            if self._codes.get(id(code)) is not code or sys._getframe(1).f_code is not code:
                self._invalidate('caller_frame_changed'); return
            if len(self.records) >= self.max_events:
                self._invalidate('budget'); return
            missing = arg0 is self.monitor.MISSING
            record = dict(index=len(self.records), thread_id=threading.get_ident(),
                caller=site(code, offset), target=target_metadata(callable_object),
                arg0_missing=missing, arg0_type=None if missing else _type_metadata(type(arg0)))
            self.records.append(record)
            if self.on_before_call is not None:
                self.on_before_call(code, offset, callable_object, arg0)
            if len(self.records) >= self.max_events:
                self._invalidate('budget')
        except BaseException as error:
            self._error('callback', error); self._invalidate('callback_error')
        finally:
            self._busy = False

    def close(self):
        if self.closed:
            return
        self._invalidate('close'); self.closed = True
        try:
            if self._owns_slot() and self.monitor.get_events(self.tool_id) == 0:
                masks = [self.monitor.get_local_events(self.tool_id, code) for code in self.codes]
                if any(mask != 0 for mask in masks):
                    self._error('close', 'CHANGED_LOCAL_MASK_NOT_RELEASED')
                    return
                previous = self.monitor.register_callback(self.tool_id, self.monitor.events.CALL, None)
                if previous is not self._callback_ref and not (previous is None and not self._callback_registered):
                    # A same-ID callback replacement has no read-only query.
                    # Preserve the discovered foreign callback and do not free
                    # its slot, rather than silently clearing it at teardown.
                    self.monitor.register_callback(self.tool_id, self.monitor.events.CALL, previous)
                    self._error('close', 'CALLBACK_CHANGED_NOT_RELEASED')
                    return
                self.callback_identity_verified_on_close = previous is self._callback_ref
                self.monitor.free_tool_id(self.tool_id)
        except BaseException as error:
            self._error('close', error)
        finally:
            if self.tool_id is not None and _OWNERS.get(self.tool_id) is self:
                del _OWNERS[self.tool_id]

    def document(self):
        return dict(schema='scar.local-call-routes', schema_version=1,
            process_id=self.pid, status=self.status, stop_reason=self.stop_reason,
            closed=self.closed, active=self.active, permanent_disable=self.permanent_disable,
            max_events=self.max_events, records=list(self.records), errors=list(self.errors),
            selected_codes=list(self._descriptions), local_event='CALL', global_events_enabled=False,
            profile_hook_installed=False, audit_hook_installed=False,
            array_values_collected=False, arg0_values_collected=False,
            automatic_callee_body_monitoring=False,
            call_attempts_only=True, entry_or_completion_observed=False,
            all_writes_captured=False, per_code_events_are_per_instance=False,
            tool_object_registry_used=True, callback_identity_verified_on_close=self.callback_identity_verified_on_close,
            active_callback_replacement_detection=False,
            no_foreign_monitoring_api_mutation_required=True,
            source_function_environment_or_effects_proven=False, deployment_authorized=False)

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
        return False


__all__ = ['LocalCallRoutes']

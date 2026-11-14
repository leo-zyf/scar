"""Selective, bounded input metadata at an original plan's code boundary.

Only exec-audited exact code objects receive local PY_START events. No global
call hook, program execution, native plan loading or tensor content collection
is performed here. Collector failures leave the original program running.
"""
import atexit
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import re
import sys
from types import CodeType

from scar.compiler.input_metadata import InputMetadataObserver
from scar.compiler.model import read_plan
from scar.trace.code_snapshot import code_key


_installed=None
_RESUMABLE=(inspect.CO_GENERATOR|inspect.CO_COROUTINE|inspect.CO_ASYNC_GENERATOR
    |inspect.CO_ITERABLE_COROUTINE)


class RegionMetadataCollector:
    def __init__(self,plan_path,out):
        self.plan_path=Path(plan_path).absolute() if plan_path else None
        self.out=Path(out).absolute() if out else None
        self.pid=os.getpid();self.observer=InputMetadataObserver(max_calls_per_code=2)
        self.plan=None;self.source=None;self.parameters=();self.monitor=None;self.tool_id=None
        self.tool_name='scar.region-metadata.'+str(self.pid)+'.'+str(id(self))
        self.codes={};self.roots={};self.errors=[];self.matched=0;self.callbacks=0
        self.exec_checks=0;self.hook_installed=False;self._marker=object()
        self.closed=False;self.stopped=False;self._busy=False;self.status='WAITING'
        self.window_flushed=False;self.fork_resets=0

    def _error(self,stage,error):
        record=dict(stage=stage,error=error if type(error)is str else type(error).__name__)
        if record not in self.errors and len(self.errors)<32:self.errors.append(record)

    def _owns_tool(self):
        return self.tool_id is not None and self.monitor.get_tool(self.tool_id)==self.tool_name

    def start(self):
        try:
            if self.plan_path is None or self.out is None:raise ValueError('plan and output directory required')
            self.plan=read_plan(self.plan_path);self.source=self.plan.region.source
            self.parameters=self.plan.region.parameters
            if (not Path(self.source.path).is_absolute()
                    or re.fullmatch(r'[0-9a-f]{64}',self.source.sha256) is None
                    or re.fullmatch(r'pycode-v2:[0-9a-f]{64}',self.source.code_key) is None
                    or len(set(self.parameters))!=len(self.parameters)):
                raise ValueError('exact source/code identity and unique parameters required')
            self.source_path=str(Path(self.source.path).resolve())
            self.monitor=getattr(sys,'monitoring',None)
            if self.monitor is None:
                self.status='UNAVAILABLE';self.stopped=True
                self._error('monitoring','SYS_MONITORING_UNAVAILABLE');self.flush();return self
            for tool in range(5,-1,-1):
                if self.monitor.get_tool(tool) is not None:continue
                try:self.monitor.use_tool_id(tool,self.tool_name)
                except ValueError:continue
                self.tool_id=tool;break
            if self.tool_id is None:
                self.status='UNAVAILABLE';self.stopped=True
                self._error('monitoring','NO_FREE_MONITORING_TOOL_ID');self.flush();return self
            # A newly claimed tool starts with zero global events. Never call
            # set_events: local target events are the entire observation scope.
            if self.monitor.get_events(self.tool_id)!=0:raise RuntimeError('nonzero global monitoring events')
            self.monitor.register_callback(self.tool_id,self.monitor.events.PY_START,self._on_start)
            sys.addaudithook(self._audit)
            sys.audit('scar.region_metadata.install',self._marker)
            if not self.hook_installed:raise RuntimeError('exec audit hook was not installed')
            if hasattr(os,'register_at_fork'):os.register_at_fork(after_in_child=self._after_fork)
        except Exception as error:
            self._error('install',error);self.status='ERROR';self.stopped=True
            self._disable();self.flush()
        return self

    def _after_fork(self):
        try:
            self.pid=os.getpid();self.observer=InputMetadataObserver(max_calls_per_code=2)
            self.callbacks=0;self.errors=[];self.closed=False;self.stopped=False;self._busy=False
            self.window_flushed=False;self.fork_resets+=1;self.status='WAITING'
            if self._owns_tool():
                for code in self.codes.values():
                    self.monitor.set_local_events(self.tool_id,code,self.monitor.events.PY_START)
                if self.codes:self.status='ARMED'
            else:
                self.status='UNAVAILABLE';self.stopped=True
                self._error('fork','MONITORING_TOOL_NOT_INHERITED')
        except Exception as error:
            self._error('fork',error);self.status='ERROR';self.stopped=True

    def _audit(self,event,args):
        try:
            if os.getpid()!=self.pid:self._after_fork()
            if event=='scar.region_metadata.install' and args and args[0]is self._marker:
                self.hook_installed=True;return
            if self.closed or self.stopped or event!='exec' or not args or type(args[0])is not CodeType:return
            root=args[0]
            if str(Path(root.co_filename).resolve())!=self.source_path or id(root) in self.roots:return
            self.roots[id(root)]=root;self.exec_checks+=1
            if sha256(Path(self.source.path).read_bytes()).hexdigest()!=self.source.sha256:
                self._error('exec','SOURCE_SHA256_MISMATCH');self.status='REJECTED';self.stopped=True
                self._disable();self.flush();return
            pending=[root];visited=set()
            while pending:
                code=pending.pop()
                if id(code) in visited:continue
                visited.add(id(code))
                if len(visited)>10000:raise ValueError('target code-tree budget exceeded')
                pending.extend(value for value in code.co_consts if type(value)is CodeType)
                if code.co_qualname!=self.source.qualname:continue
                if str(Path(code.co_filename).resolve())!=self.source_path:continue
                if code_key(code)!=self.source.code_key:
                    self._error('exec','CODE_ID_MISMATCH');continue
                if code.co_flags&_RESUMABLE:
                    self._error('exec','RESUMABLE_CODE_UNSUPPORTED');self.status='UNSUPPORTED';continue
                names=code.co_varnames[:code.co_argcount+code.co_kwonlyargcount]
                if code.co_flags&(inspect.CO_VARARGS|inspect.CO_VARKEYWORDS) or tuple(names)!=self.parameters:
                    self._error('exec','PARAMETER_LAYOUT_MISMATCH');self.status='REJECTED';continue
                if not self._owns_tool() or self.monitor.get_events(self.tool_id)!=0:
                    raise RuntimeError('monitoring tool or global event mask changed')
                if id(code) not in self.codes:
                    self.codes[id(code)]=code;self.matched+=1
                    self.monitor.set_local_events(self.tool_id,code,self.monitor.events.PY_START)
                    self.status='ARMED'
        except Exception as error:
            self._error('exec',error);self.status='ERROR';self.stopped=True
            self._disable();self.flush()

    def _on_start(self,code,offset):
        try:
            if os.getpid()!=self.pid:self._after_fork()
            if self.closed or self.stopped or self._busy or self.codes.get(id(code))is not code:return
            self._busy=True
            if not self._owns_tool() or self.monitor.get_events(self.tool_id)!=0:
                raise RuntimeError('monitoring tool or global event mask changed')
            # PY_START places the entered callee directly below its callback.
            # The code identity check keeps another stack frame from donating
            # parameters if an implementation changes this documented event.
            frame=sys._getframe(1)
            if frame.f_code is not code:raise RuntimeError('PY_START frame does not match target code')
            values=tuple(frame.f_locals[name] for name in self.parameters)
            self.callbacks+=1
            more=self.observer(self.source,self.parameters,values)
            if not more or self.callbacks>=2:
                self.status='COMPLETE';self.stopped=True;self._disable()
                self.window_flushed=True
                if not self.flush():self.window_flushed=False
        except Exception as error:
            self._error('callback',error);self.status='ERROR';self.stopped=True
            self._disable();self.flush()
        finally:
            self._busy=False

    def _disable(self):
        try:
            if self._owns_tool():
                for code in self.codes.values():self.monitor.set_local_events(self.tool_id,code,0)
        except Exception as error:self._error('disable',error)

    def document(self):
        result=self.observer.document()
        result.update(status=self.status,matched=self.matched,errors=list(self.errors),collector=dict(
            status=self.status,process_id=self.pid,plan=None if self.plan_path is None else str(self.plan_path),
            matched=self.matched,callbacks=self.callbacks,exec_source_hash_checks=self.exec_checks,
            local_event='PY_START',global_events_enabled=False,profile_hook_installed=False,
            max_calls=2,window_flushed=self.window_flushed,closed=self.closed,
            audit_hook_installed=self.hook_installed,resumable_code_supported=False,
            fork_observer_resets=self.fork_resets))
        return result

    def flush(self):
        try:
            if self.out is None:return False
            self.out.mkdir(parents=True,exist_ok=True)
            path=self.out/('metadata.'+str(self.pid)+'.json');temporary=path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(self.document(),indent=2,allow_nan=False)+'\n')
            os.replace(temporary,path);return True
        except Exception as error:
            self._error('write',error);return False

    def close(self):
        try:
            if os.getpid()!=self.pid:self._after_fork()
            if self.closed:return
            self.closed=True;self.stopped=True;self._disable()
            if self.status in ('WAITING','ARMED'):self.status='PARTIAL' if self.observer.records else 'NO_MATCH'
            if self._owns_tool():
                self.monitor.register_callback(self.tool_id,self.monitor.events.PY_START,None)
                self.monitor.free_tool_id(self.tool_id)
            self.flush()
        except Exception as error:
            self._error('close',error);self.flush()


def install(plan_path=None,out=None):
    """Env opt-in, idempotent per process; collector errors are isolated."""
    global _installed
    try:
        if _installed is not None and _installed.pid==os.getpid() and not _installed.closed:return _installed
        plan_path=plan_path or os.environ.get('SCAR_REGION_METADATA_PLAN')
        out=out or os.environ.get('SCAR_REGION_METADATA_OUT')
        if not plan_path and not out:return None
        collector=RegionMetadataCollector(plan_path,out);_installed=collector
        atexit.register(collector.close);return collector.start()
    except Exception:
        return None


__all__=['RegionMetadataCollector','install']

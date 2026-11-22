"""Local CALL-only target metadata for a bounded original F4 site selector.

No profile hook, global events, operand inspection or target-body monitoring.
The actual callable supplied by monitoring is described without user getters.
"""
import atexit
from hashlib import sha256
import inspect,json,os,sys,threading,types
from pathlib import Path

from scar.compiler.call_binding import read_call_binding_plan,plan_digest,OBSERVATION_SCHEMA
from scar.trace.code_snapshot import code_key, describe_code
from scar.trace.lean_capture import LeanCapturePlan


_installed=None
_RESUMABLE=inspect.CO_GENERATOR|inspect.CO_COROUTINE|inspect.CO_ASYNC_GENERATOR


def _type_metadata(cls):
    result={}
    for key in ('__module__','__qualname__'):
        value=type.__dict__[key].__get__(cls)
        result[key.strip('_')]=value if type(value)is str else None
    return result


def target_metadata(value):
    """Observe exact builtin carriers, without repr/str or dynamic getattr."""
    kind=type(value);result=dict(kind='OTHER_CALLABLE',type=_type_metadata(kind),code=None,
        native_boundary=dict(status='NATIVE_SCHEMA_AND_IMPLEMENTATION_UNRESOLVED'))
    function=None
    if kind is types.FunctionType:
        result['kind']='PYTHON_FUNCTION';function=value
    elif kind is types.MethodType:
        result['kind']='BOUND_PYTHON_METHOD';function=value.__func__
        result['receiver_type']=_type_metadata(type(value.__self__))
    elif kind in (types.BuiltinFunctionType,types.BuiltinMethodType,types.MethodWrapperType,
                  types.MethodDescriptorType,types.WrapperDescriptorType):
        result['kind']='NATIVE_CARRIER'
        fields={}
        for key in ('__module__','__name__','__qualname__'):
            # These exact builtin types have no application __getattribute__.
            item=getattr(value,key,None)
            if type(item)is str:fields[key.strip('_')]=item
        result['native_boundary']=dict(status='EXACT_CARRIER_OBSERVED_SCHEMA_NOT_JOINED',**fields)
    else:
        from scar.trace.python_call_targets import _is_type_object
        if _is_type_object(value):result['kind']='TYPE_OBJECT';result['constructed_type']=_type_metadata(value)
    if type(function)is types.FunctionType:
        result['code']=describe_code(function.__code__)
        result['native_boundary']=dict(status='PYTHON_CODE_OBSERVED_TRANSITIVE_NATIVE_CALLS_NOT_CAPTURED')
    return result


class CallBindingCollector:
    def __init__(self,plan_path,out):
        self.plan_path=Path(plan_path).absolute() if plan_path else None;self.out=Path(out).absolute() if out else None
        self.pid=os.getpid();self.plan=None;self.selector=None;self.monitor=getattr(sys,'monitoring',None)
        self.tool_id=None;self.tool_name='scar.call-binding.'+str(self.pid)+'.'+str(id(self))
        self.codes={};self.roots={};self.verified_sources=set();self.sites={};self.counts={};self.records=[];self.errors=[]
        self.status='WAITING';self.closed=False;self.stopped=False;self.busy=False;self.matched=0;self.callbacks=0
        self.hook_installed=False;self.marker=object()

    def _error(self,stage,error):
        if len(self.errors)<32:self.errors.append(dict(stage=stage,error=error if type(error)is str else type(error).__name__))

    def _owns(self):return self.tool_id is not None and self.monitor.get_tool(self.tool_id)==self.tool_name

    def start(self):
        try:
            self.plan=read_call_binding_plan(self.plan_path)
            self.sites={(s['code_key'],s['offset']):s for s in self.plan['sites']};self.counts={s['id']:0 for s in self.plan['sites']}
            if not self.sites:self.status='UNSUPPORTED';self.stopped=True;self.flush();return self
            self.selector=LeanCapturePlan.from_dict(self.plan['selector'],max_codes=64)
            if self.monitor is None:self.status='UNAVAILABLE';self.stopped=True;self._error('start','SYS_MONITORING_UNAVAILABLE');self.flush();return self
            if sys.implementation.name!='cpython' or sys.version_info[:3]!=(3,14,7):
                self.status='UNSUPPORTED';self.stopped=True;self._error('start','PINNED_CPYTHON_3_14_7_CALL_CONTRACT_REQUIRED');self.flush();return self
            for tool in range(5,-1,-1):
                if self.monitor.get_tool(tool)is not None:continue
                try:self.monitor.use_tool_id(tool,self.tool_name)
                except ValueError:continue
                self.tool_id=tool;break
            if self.tool_id is None:raise RuntimeError('no free monitoring tool ID')
            if self.monitor.get_events(self.tool_id)!=0:raise RuntimeError('nonzero global monitoring mask')
            self.monitor.register_callback(self.tool_id,self.monitor.events.CALL,self._on_call)
            sys.addaudithook(self._audit);sys.audit('scar.call_binding.install',self.marker)
            if not self.hook_installed:raise RuntimeError('exec audit hook not installed')
            if hasattr(os,'register_at_fork'):os.register_at_fork(after_in_child=self._after_fork)
        except Exception as error:
            self._error('start',error);self.status='ERROR';self.stopped=True;self._disable();self.flush()
        return self

    def _after_fork(self):
        # Without an original child activation selector, inherited parent
        # observations cannot be presented as child CALL attempts.
        self.pid=os.getpid();self.records=[];self.counts={s['id']:0 for s in self.plan['sites']} if self.plan else {}
        self.errors=[];self.status='UNSUPPORTED_FORK';self.stopped=True;self.closed=False;self._disable()
        self._error('fork','CHILD_CAPTURE_NOT_ARMED');self.flush()

    def _audit(self,event,args):
        try:
            if event=='scar.call_binding.install' and args and args[0]is self.marker:self.hook_installed=True;return
            if self.closed or self.stopped or event!='exec' or not args or type(args[0])is not types.CodeType:return
            root=args[0]
            names={Path(body.source.path).name for body in self.selector.bodies}
            if Path(root.co_filename).name not in names:return
            path=str(Path(root.co_filename).resolve())
            if path not in {b.source.path for b in self.selector.bodies} or id(root) in self.roots:return
            if len(self.roots)>=64:raise ValueError('matching exec-root budget exceeded')
            self.roots[id(root)]=root
            self.selector.verify_sources();self.verified_sources.add(path)
            pending=[root];visited=set()
            while pending:
                code=pending.pop()
                if id(code) in visited:continue
                visited.add(id(code))
                if len(visited)>10000:raise ValueError('code-tree budget exceeded')
                pending.extend(x for x in code.co_consts if type(x)is types.CodeType)
                candidate=self.selector.candidate(code)
                if candidate is None:continue
                if self.selector.match(code)is None:
                    self._error('exec','CODE_ID_MISMATCH');continue
                if code.co_flags&_RESUMABLE:
                    self._error('exec','RESUMABLE_CODE_UNSUPPORTED');continue
                if not self._owns() or self.monitor.get_events(self.tool_id)!=0:raise RuntimeError('monitoring tool changed')
                if id(code) not in self.codes:
                    self.codes[id(code)]=code;self.matched+=1
                    self.monitor.set_local_events(self.tool_id,code,self.monitor.events.CALL);self.status='ARMED'
        except Exception as error:
            self._error('exec',error);self.status='ERROR';self.stopped=True;self._disable();self.flush()

    def _on_call(self,code,offset,callable_object,arg0):
        try:
            if self.closed or self.stopped or os.getpid()!=self.pid:return self.monitor.DISABLE
            if self.busy or self.codes.get(id(code))is not code:return
            self.busy=True;self.callbacks+=1
            if not self._owns() or self.monitor.get_events(self.tool_id)!=0:raise RuntimeError('monitoring tool changed')
            if sys._getframe(1).f_code is not code:raise RuntimeError('CALL frame differs from target caller')
            key=code_key(code),offset;selected=self.sites.get(key)
            if selected is None:return self.monitor.DISABLE
            identity=selected['id'];limit=self.plan['max_targets_per_site']
            if self.counts[identity]>=limit:return self.monitor.DISABLE
            self.records.append(dict(index=len(self.records)+1,site_id=identity,caller_code_key=key[0],offset=offset,
                thread_id=threading.get_ident(),target=target_metadata(callable_object)))
            self.counts[identity]+=1
            if all(n>=limit for n in self.counts.values()):
                self.status='COMPLETE';self.stopped=True;self._disable();self.flush()
            if self.counts[identity]>=limit:return self.monitor.DISABLE
        except Exception as error:
            self._error('callback',error);self.status='ERROR';self.stopped=True;self._disable();self.flush()
        finally:self.busy=False

    def _disable(self):
        try:
            if self._owns():
                for code in self.codes.values():self.monitor.set_local_events(self.tool_id,code,0)
        except Exception as error:self._error('disable',error)

    def document(self):
        return dict(schema=OBSERVATION_SCHEMA,schema_version=1,plan_sha256=None if self.plan is None else plan_digest(self.plan),
            process_id=self.pid,status=self.status,records=list(self.records),site_counts=dict(self.counts),errors=list(self.errors),
            collector=dict(local_event='CALL',global_events_enabled=False,profile_hook_installed=False,
                selected_codes=self.matched,callbacks=self.callbacks,max_targets_per_site=None if self.plan is None else self.plan['max_targets_per_site'],
                source_files_sha_verified=sorted(self.verified_sources),audit_hook_installed=self.hook_installed,closed=self.closed,
                array_values_collected=False,callee_bodies_monitored=False,entry_or_completion_observed=False,resumable_code_supported=False))

    def flush(self):
        try:
            if self.out is None:return False
            self.out.mkdir(parents=True,exist_ok=True);path=self.out/('call-bindings.'+str(self.pid)+'.json');tmp=path.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(self.document(),indent=2,allow_nan=False)+'\n');os.replace(tmp,path);return True
        except Exception as error:self._error('write',error);return False

    def close(self):
        try:
            if self.closed:return
            self.closed=True;self.stopped=True;self._disable()
            if self.status in ('WAITING','ARMED'):self.status='PARTIAL' if self.records else 'NO_MATCH'
            if self._owns():self.monitor.register_callback(self.tool_id,self.monitor.events.CALL,None);self.monitor.free_tool_id(self.tool_id)
            self.flush()
        except Exception as error:self._error('close',error);self.flush()


def install(plan_path=None,out=None):
    """Idempotent env opt-in; optional collector errors never exit the target."""
    global _installed
    try:
        if _installed is not None and _installed.pid==os.getpid() and not _installed.closed:return _installed
        plan_path=plan_path or os.environ.get('SCAR_CALL_BINDING_PLAN');out=out or os.environ.get('SCAR_CALL_BINDING_OUT')
        if not plan_path and not out:return None
        _installed=CallBindingCollector(plan_path,out);atexit.register(_installed.close);return _installed.start()
    except Exception:return None


__all__=['CallBindingCollector','target_metadata','install']

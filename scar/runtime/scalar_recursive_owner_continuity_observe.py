"""Observe the exact original inter-search owner window in CPython."""

import argparse
import builtins
from collections import Counter
import os
from pathlib import Path
import sys
from threading import get_ident
from types import BuiltinMethodType, CodeType, FunctionType

from scar.analysis.scalar_recursive_owner_continuity import CAPTURE, DESIGN, _require
from scar.analysis.scalar_table_handoff import _ref, _write
from scar.ir.execution_recipe import read_artifact
from scar.runtime.entry_runner import run_original_script
from scar.runtime.scalar_recursive_driver_observe import ordered_map_digest
from scar.runtime.scalar_recursive_observe import _receiver_storage
from scar.runtime.source_value_observe import _codes, _coordinate, _target_coordinate
from scar.trace.code_snapshot import describe_code, site

_HASATTR, _LEN = builtins.hasattr,builtins.len
_PROVEN = {'KEYED_GET','LENGTH_READ','PRESENCE_TEST'}


class _OwnerWindow:
    def __init__(self,design):
        self.design = design
        self.static = design['static']
        self.monitor,self.events = sys.monitoring,sys.monitoring.events
        self.slot = next(i for i in range(5,-1,-1) if self.monitor.get_tool(i) is None)
        self.active = True
        self.sequence = 0
        self.tokens,self.windows,self.objects = [],[],[]
        self.frontiers,self.pending = Counter(),{}
        self.current,self.owner = None,None
        self.probe_owner = read_artifact(design['source_driver'])['probe']['source_recursive_owner']
        self.wants = {_target_coordinate(code):(name,code) for name,code in self.static['scopes'].items()}
        self.wants.update({_target_coordinate(code):(role,code) for role,code in
            (('@driver',self.static['driver']['code']),('@probe',self.probe_owner))})
        self.codes = {}
        self.access = {(row['owner'],row['site']['offset']):row for row in self.static['access_sites']}

    def tick(self):
        self.sequence += 1
        return self.sequence

    def identity(self,obj):
        for ordinal,value in enumerate(self.objects,1):
            if obj is value:
                return ordinal
        _require(len(self.objects)<16,'original owner retention observation limit exceeded')
        self.objects.append(obj)
        return len(self.objects)

    def attach(self,key,args):
        if (not self.active or key!='exec' or not args or type(args[0]) is not CodeType
                or args[0].co_filename not in (self.design['source']['uri'],self.design['ui_source']['uri'])):
            return
        for code in _codes(args[0]):
            selected = self.wants.get(_coordinate(code))
            if selected is None:
                continue
            role,descriptor = selected
            _require(describe_code(code)==descriptor,'original owner-window body was replaced')
            self.codes[code] = role
            mask = self.events.PY_START|self.events.PY_RETURN
            if role!='@probe':
                mask |= self.events.INSTRUCTION
            if any(owner==role for owner,_ in self.access):
                mask |= self.events.CALL
            self.monitor.set_local_events(self.slot,code,mask)

    def holders(self):
        storage = _receiver_storage(self.owner)
        rows = {}
        for role,field in self.design['roles'].items():
            value = storage[field]
            _require(type(value) is dict,'original owner holder is not exact dictionary')
            rows[role] = dict(identity=self.identity(value),size=len(value),ordered_sha256=ordered_map_digest(value))
        return rows

    def _anchor(self,name,frame,offset,*,holders=False,**extras):
        anchors = self.current['anchors']
        _require(name not in anchors,'duplicate original lifecycle transition')
        row = dict(sequence=self.tick(),thread_id=get_ident(),site=site(frame.f_code,offset),
                   actor=self.identity(self.owner),**extras)
        if holders:
            row['holders'] = self.holders()
        anchors[name] = row

    def unwind(self,code,offset,exception):
        if self.codes.get(code)!='@driver' or type(exception) is not GeneratorExit or self.windows or self.current is not None:
            return
        frame = sys._getframe(1)
        self.owner = frame.f_locals[self.static['driver']['receiver']]
        self.current = dict(owner_identity=self.identity(self.owner),anchors={},events=[])
        self._anchor('close',frame,offset,holders=True,exception='GeneratorExit')

    def start(self,code,offset):
        if self.current is None:
            return
        frame = sys._getframe(1)
        role = self.codes[code]
        anchors = self.current['anchors']
        if role==self.static['source_consumer'] and 'submit' in anchors and 'consumer_start' not in anchors:
            owner = frame.f_locals[self.static['owner_formals'][role]]
            _require(owner is self.owner,'next original consumer has another owner')
            self._anchor('consumer_start',frame,offset)
        elif role=='@driver' and 'consumer_start' in anchors and 'driver_start' not in anchors:
            _require(frame.f_locals[self.static['driver']['receiver']] is self.owner,'original driver receiver changed')
            self._anchor('driver_start',frame,offset,holders=True)
        elif role=='@probe' and 'fresh_score_reset' in anchors:
            parent = frame.f_back
            if parent is None or self.codes.get(parent.f_code)!='@driver' or parent.f_lasti!=self.static['probe']['offset']:
                return
            _require(frame.f_locals[self.probe_owner['parameters'][0]] is self.owner,
                     'first original recursive probe did not receive the retained owner')
            self._anchor('first_probe',frame,offset,holders=True,caller=site(parent.f_code,parent.f_lasti))
            self.windows.append(self.current)
            self.current = None
            self.pending.clear()

    def finish(self,code,offset,_value):
        if (self.current is None or self.codes[code]!=self.static['source_consumer']
                or 'worker_return' in self.current['anchors']):
            return
        frame = sys._getframe(1)
        _require(frame.f_locals[self.static['owner_formals'][self.codes[code]]] is self.owner,
                 'returning original worker has another owner')
        self._anchor('worker_return',frame,offset)

    def instruction(self,code,offset):
        role = self.codes[code]
        launcher = self.static['launcher']
        is_input = role==launcher['name'] and offset==launcher['command']['after']['offset']
        is_submit = role==launcher['name'] and offset==launcher['submit']['offset']
        is_reset = role=='@driver' and offset==self.static['reset']['after']['offset']
        candidate = self.access.get((role,offset))
        is_access = candidate is not None and candidate['site']['opname'] not in ('CALL','CALL_KW','CALL_FUNCTION_EX')
        if not (is_input or is_submit or is_reset or is_access):
            return self.monitor.DISABLE
        frame = sys._getframe(1)
        if is_input:
            args = frame.f_locals[launcher['command']['symbol']]
            _require(type(args) is list and args and all(type(word) is str for word in args) and len(self.tokens)<16,
                     'original input tokenization left the finite command contract')
            self.tokens.append(list(args))
            return
        if self.current is None:
            return
        if is_submit and 'submit' not in self.current['anchors']:
            target = frame.f_locals[launcher['callable_local']]
            desc = self.static['scopes'][self.static['source_consumer']]
            _require(type(target) is FunctionType and describe_code(target.__code__)==desc
                and frame.f_locals[launcher['receiver']] is self.owner,'executor submission did not retain the proven owner and consumer')
            self._anchor('submit',frame,offset,holders=True,input_ordinal=len(self.tokens)-1,target=desc)
        elif is_reset and 'driver_start' in self.current['anchors'] and 'fresh_score_reset' not in self.current['anchors']:
            self._anchor('fresh_score_reset',frame,offset,holders=True,write=self.static['reset']['site'])
        elif is_access:
            self._event(frame,candidate,False,None)

    def call(self,code,offset,target,arg0):
        row = self.access.get((self.codes[code],offset))
        if row is None:
            return self.monitor.DISABLE
        if self.current is None:
            return
        frame = sys._getframe(1)
        if row['kind'] not in _PROVEN or row['receiver'] is None or frame.f_locals[row['receiver']] is not self.owner:
            self._event(frame,row,False,None)
            return
        root = _receiver_storage(self.owner)[self.design['roles'][row['role']]]
        if row['kind']=='KEYED_GET':
            exact = target is dict.get and arg0 is root
            if type(target) is BuiltinMethodType:
                exact = target.__self__ is root and target.__name__=='get'
        elif row['kind']=='PRESENCE_TEST':
            exact = target is _HASATTR and arg0 is self.owner
        else:
            exact = target is _LEN and arg0 is root
        if exact and type(root) is dict:
            self.pending[frame,offset] = (row,root)
        else:
            self._event(frame,row,False,None)

    def _event(self,frame,row,exact,root):
        events = self.current['events']
        _require(len(events)<128,'inter-search observation exceeded the selected event budget')
        events.append(dict(owner=row['owner'],site=row['site'],role=row['role'],kind=row['kind'],
            sequence=self.tick(),thread_id=get_ident(),exact_normal_builtin=exact,
            table_identity=self.identity(root) if exact else None))

    def c_return(self,code,offset,_target,_arg0):
        self._complete_builtin(offset,True,sys._getframe(1))

    def c_raise(self,code,offset,_target,_arg0):
        self._complete_builtin(offset,False,sys._getframe(1))

    def _complete_builtin(self,offset,normal,frame):
        selected = self.pending.pop((frame,offset),None)
        if self.current is not None and selected is not None:
            row,root = selected
            unchanged = _receiver_storage(self.owner)[self.design['roles'][row['role']]] is root
            self._event(frame,row,normal and unchanged,root)

    def close(self):
        self.active = False
        self.monitor.set_events(self.slot,0)
        for code in self.codes:
            self.monitor.set_local_events(self.slot,code,0)


def capture(design_path, stdin, out):
    ref = _ref(design_path,DESIGN)
    plan = read_artifact(ref)
    read_artifact(plan['source'],json_document=False)
    read_artifact(plan['ui_source'],json_document=False)
    stream = _ref(stdin,'application/octet-stream')
    read_artifact(stream,json_document=False)
    watch = _OwnerWindow(plan)
    monitor,events,slot = watch.monitor,watch.events,watch.slot
    callbacks = ((events.PY_START,watch.start),(events.PY_RETURN,watch.finish),(events.PY_UNWIND,watch.unwind),
                 (events.INSTRUCTION,watch.instruction),(events.CALL,watch.call),
                 (events.C_RETURN,watch.c_return),(events.C_RAISE,watch.c_raise))
    monitor.use_tool_id(slot,CAPTURE)
    try:
        for event,fn in callbacks:
            monitor.register_callback(slot,event,fn)
        monitor.set_events(slot,events.PY_UNWIND)
        sys.addaudithook(watch.attach)
        launch = run_original_script(plan['source']['uri'],[])
    finally:
        watch.close()
        for event,_ in callbacks:
            monitor.register_callback(slot,event,None)
        monitor.free_tool_id(slot)
    if watch.current is not None:
        watch.frontiers['INCOMPLETE_OWNER_WINDOW'] += 1
    return _write(out,dict(schema=CAPTURE,schema_version=1,design=ref,source=plan['source'],launch=launch,
        stdin=stream,input_tokens=watch.tokens,windows=watch.windows,frontiers=dict(watch.frontiers),
        process_id=os.getpid(),original_execution=True,performance_measurement=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',type=Path,required=True)
    parser.add_argument('--stdin',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    capture(args.design,args.stdin,args.out)


if __name__=='__main__':
    main()

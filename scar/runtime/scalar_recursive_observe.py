"""Collect bounded CPython recursive calls and state edges in the original entry."""

import argparse
from collections import Counter
from dis import get_instructions
import json
import os
from pathlib import Path
import sys
from threading import get_ident
from types import BuiltinMethodType, CodeType, FunctionType, GetSetDescriptorType, MethodType

from scar.analysis.scalar_recursive_observations import CAPTURE, DESIGN, _integer, _require, ordered_digest
from scar.analysis.scalar_table_handoff import _ref, _write
from scar.ir.execution_recipe import read_artifact
from scar.runtime.entry_runner import run_original_script
from scar.runtime.scalar_observe import _state
from scar.runtime.source_value_observe import _codes, _coordinate, _instance_field, _target_coordinate
from scar.trace.code_snapshot import describe_code, site


def _receiver_storage(value):
    for base in type.__getattribute__(type(value),'__mro__'):
        descriptor = type.__getattribute__(base,'__dict__').get('__dict__')
        if descriptor is not None:
            _require(type(descriptor) is GetSetDescriptorType,'receiver must use ordinary instance storage')
            storage = descriptor.__get__(value)
            _require(type(storage) is dict,'expected exact receiver fields')
            return storage
    raise ValueError('no ordinary instance storage')


def _pod(value,depth=0):
    _require(depth <= 8,'root state exceeds the bounded nested record contract')
    if type(value) is bool or _integer(value) or type(value) is str:
        return value
    if type(value) is float:
        return {'float_hex':value.hex()}
    if value is None:
        return None
    if tuple in type.__getattribute__(type(value),'__mro__'):
        _require(tuple.__len__(value)<=16,'root state tuple exceeds bounded width')
        record = [_pod(child,depth+1) for child in tuple.__iter__(value)]
        return record if type(value) is tuple else {'type':type.__getattribute__(type(value),'__qualname__'),'fields':record}
    if type(value) in (set,frozenset):
        _require(len(value)<=256,'root state membership set exceeds bounded width')
        return sorted((_pod(child,depth+1) for child in value),key=lambda row:json.dumps(row,sort_keys=True))
    if type(value) is dict:
        _require(len(value)<=256,'root state dict exceeds bounded width')
        return [[_pod(key,depth+1),_pod(child,depth+1)] for key,child in value.items()]
    raise ValueError('non-POD source state')


class _Observe:
    def __init__(self,design,limit,state_limit):
        self.design,self.limit,self.state_limit = design,limit,state_limit
        self.monitor,self.events = sys.monitoring,sys.monitoring.events
        self.slot = next(i for i in range(5,-1,-1) if self.monitor.get_tool(i) is None)
        self.active = True
        desc = [('owner',design['owner']),('score',design['score']),('rotation',design['rotation'])]
        desc += [('transition',row['producer']) for row in design['sites'] if row['phase']=='DIRECT_TRANSITION']
        self.descriptors = {_target_coordinate(value):(role,value) for role,value in desc}
        self.roles,self.frames,self.phases,self.nested,self.ready,self.pending = {},{},{},{},{},{}
        self.call_sites = {row['site']['offset']:row for row in design['sites']}
        self.produce_sites = {row['producer_site']['offset']:row for row in design['sites']}
        self.state_at = {row['at']['offset']:row for row in design['state_sites']}
        self.state_after = {row['after']['offset']:row for row in design['state_sites'] if 'after' in row}
        self.counter = next(row['field'] for row in design['state_sites'] if row['action']=='SCALAR_STORE')
        region = read_artifact(design['region'])
        self.receiver_fields = {row['field'] for row in region['graph']['receiver_state']}
        self.dictionary_fields = sorted({row['field'] for row in design['state_sites'] if row['action']=='DICT_GET'})
        self.position = design['sites'][0]['position_formal']
        self.field_names = [row['name'] for row in design['state_fields']]
        self.roots,self.receivers = [],[]
        self.root_done,self.root_pending = set(),{}
        self.root_probes,self.rows,self.state_events = [],[],[]
        self.counts,self.state_counts,self.frontiers = Counter(),Counter(),Counter()
        self.module_i64,self.global_reads,self.code = None,None,None
        self.ordinal,self.sequence = 0,0

    def tick(self):
        self.sequence += 1
        return self.sequence

    @staticmethod
    def token(values,obj):
        for ordinal,prior in enumerate(values,1):
            if prior is obj:
                return ordinal
        values.append(obj)
        return len(values)

    def attach(self,event,args):
        if (not self.active or event!='exec' or not args or type(args[0]) is not CodeType
                or args[0].co_filename!=self.design['source']['uri']):
            return
        for code in _codes(args[0]):
            selected = self.descriptors.get(_coordinate(code))
            if selected is None:
                continue
            role,descriptor = selected
            _require(describe_code(code)==descriptor,'original selected code differs')
            self.roles[code] = role
            mask = self.events.PY_START|self.events.PY_RETURN
            if role=='owner':
                self.code = code
                self.global_reads = sorted({op.argval for op in get_instructions(code) if op.opname=='LOAD_GLOBAL'})
                mask |= self.events.CALL|self.events.INSTRUCTION
            self.monitor.set_local_events(self.slot,code,mask)

    def _actuals(self,frame,entry=None):
        local = frame.f_locals if entry is None else entry
        actuals = {key:local[key] for key in self.design['formals'] if key!=self.design['receiver']}
        actor = frame.f_globals[self.design['state_constructor']]
        actuals[self.position] = _state(actuals[self.position],actor,self.field_names)
        _require(all(type(value) is bool or _integer(value) for key,value in actuals.items() if key!=self.position),
                 'recursive input does not have exact primitive leaves')
        return actuals

    def _root_snapshot(self,receiver,position):
        root = _receiver_storage(receiver)
        state = {}
        for key,value in root.items():
            if key not in self.receiver_fields and value is not position:
                continue
            if key in self.dictionary_fields:
                _require(type(value) is dict and len(value)<=256,'root probe dictionary exceeds exact bounded capture')
                items = _pod(value)
                state[key] = dict(items=items,ordered_sha256=ordered_digest(items))
            else:
                state[key] = _pod(value)
        return state

    def _root(self,frame,receiver,activation):
        if len(self.root_probes)>=2 or len(self.root_pending)>=2:
            return
        defaults = self.design['sites'][0]['defaults']
        if len(defaults)!=1 or defaults[0]['value'] is not False or frame.f_locals[defaults[0]['formal']] is not True:
            return
        try:
            if any(type(table) is not dict or len(table) for table in
                   (_instance_field(receiver,key) for key in self.dictionary_fields)):
                return
            token = self.token(self.receivers,receiver)
            if token in self.root_done:
                return
            actuals = self._actuals(frame)
            before = self._root_snapshot(receiver,frame.f_locals[self.position])
            self.root_done.add(token)
            self.root_pending[frame] = dict(activation=activation,receiver_occurrence=token,thread_id=get_ident(),
                actuals=actuals,before=before,entry_sequence=self.tick(),module_i64={name:frame.f_globals[name]
                    for name in self.global_reads if name in frame.f_globals and _integer(frame.f_globals[name])})
        except (TypeError,ValueError,KeyError):
            self.frontiers['DRIVER_ROOT_STATE_OPEN'] += 1

    def start(self,code,_offset):
        frame = sys._getframe(1)
        parent = frame.f_back
        role = self.roles[code]
        if role=='owner':
            self.ordinal += 1
            activation = self.ordinal
            local = frame.f_locals
            receiver = local[self.design['receiver']]
            initial = {name:local[name] for name in self.design['formals']}
            self.frames[frame] = dict(id=activation,receiver=receiver,thread=get_ident(),pending=None,entry=initial)
            if self.module_i64 is None:
                self.module_i64 = {name:frame.f_globals[name] for name in self.global_reads
                    if name in frame.f_globals and _integer(frame.f_globals[name])}
            if parent is not None and parent.f_code is code:
                self._child(frame,parent,receiver,activation)
            else:
                self._root(frame,receiver,activation)
            return
        if parent is None:
            return
        try:
            owner = self.frames.get(parent)
            expected = self.produce_sites.get(parent.f_lasti) if owner else None
            if (expected is not None and self.counts[expected['index']]<self.limit
                    and describe_code(code)==expected['producer'] and site(parent.f_code,parent.f_lasti)==expected['producer_site']):
                position = parent.f_locals[self.position]
                parameter = expected['producer']['parameters'][0]
                _require(frame.f_locals[parameter] is position,'phase did not use same original parent state actual')
                details = dict(kind=expected['phase'],site=expected['producer_site'],score_observed=False)
                if role=='rotation':
                    argument = expected['producer']['parameters'][1]
                    details['rotation_flag'] = frame.f_locals[argument]
                self.phases[frame] = (parent,expected,details)
                return
            phase = self.phases.get(parent)
            if phase is not None and role=='score' and parent.f_lasti==self.design['score_call']['offset']:
                source_formals = phase[1]['producer']['parameters']
                _require(all(frame.f_locals[callee] is parent.f_locals[caller] for callee,caller in
                    zip(self.design['score']['parameters'],source_formals,strict=True)),
                    'nested scorer did not bind the original transition operands')
                self.nested[frame] = parent
        except (TypeError,ValueError,KeyError):
            self.frontiers['ORIGINAL_PHASE_INPUT_OPEN'] += 1

    def _child(self,frame,parent,receiver,activation):
        key = parent,parent.f_lasti
        selected = self.call_sites.get(parent.f_lasti)
        admitted = self.pending.pop(key,None)
        if selected is None or self.counts[selected['index']]>=self.limit or admitted is None:
            return
        try:
            owner,typed,phase,result,produced,called = admitted
            _require(owner['receiver'] is receiver and owner['thread']==get_ident()
                and frame.f_locals[self.position] is result, 'recursive child changed actual identity or receiver')
            actual = self._actuals(frame)
            _require(all(actual[entry['formal']]==entry['value'] for entry in selected['defaults']),
                     'recursive default did not bind as declared')
            child = self.frames[frame]
            child['pending'] = dict(site_index=selected['index'],site=selected['site'],phase=phase,
                parent_frame=owner['id'],child_frame=activation,thread_id=get_ident(),parent_entry=self._actuals(parent,owner['entry']),
                parent_actual=typed,child_actual=actual,same_receiver=True,same_produced_actual_object=True,
                produce_sequence=produced,call_sequence=called,child_enter_sequence=self.tick(),
                state=dict(entry_counter=_instance_field(receiver,self.counter)))
        except (TypeError,ValueError,KeyError):
            self.frontiers['ORIGINAL_CHILD_ACTUAL_OPEN'] += 1

    def call(self,code,offset,target,arg0):
        frame = sys._getframe(1)
        owner = self.frames.get(frame)
        if owner is None:
            return
        selected = self.call_sites.get(offset)
        if selected is not None:
            if self.counts[selected['index']]>=self.limit:
                return self.monitor.DISABLE
            recent = self.ready.pop(frame,None)
            if recent is None:
                return
            try:
                if type(target) is MethodType:
                    bound = target.__self__ is owner['receiver']
                    function = target.__func__
                else:
                    bound = arg0 is owner['receiver']
                    function = target
                _require(bound and type(function) is FunctionType and function.__code__ is self.code
                    and site(code,offset)==selected['site'] and selected is recent[0],
                    'actual recursive callable differs from original owner')
                _,phase,result,produced = recent
                self.pending[frame,offset] = (owner,self._actuals(frame),phase,result,produced,self.tick())
            except (TypeError,ValueError,KeyError):
                self.frontiers['ORIGINAL_RECURSIVE_DISPATCH_OPEN'] += 1
            return
        state = self.state_at.get(offset)
        if state is None or state['action']!='DICT_GET':
            return self.monitor.DISABLE
        if self.state_counts[offset]>=self.state_limit:
            return self.monitor.DISABLE
        try:
            receiver = owner['receiver']
            root = _instance_field(receiver,state['field'])
            exact = target is dict.get and arg0 is root
            if type(target) is BuiltinMethodType:
                exact = target.__self__ is root and target.__name__=='get'
            _require(exact and type(root) is dict,'original state query is not exact builtin dict.get')
            self.pending[frame,offset] = (self.tick(),root,len(root))
        except (TypeError,ValueError,KeyError):
            self.frontiers['ORIGINAL_DICT_QUERY_OPEN'] += 1

    def c_return(self,code,offset,_target,_arg0):
        frame = sys._getframe(1)
        state = self.state_at.get(offset)
        if state is not None and state['action']=='DICT_GET':
            self._commit(frame,state,'ORIGINAL_BUILTIN_RETURN')

    def instruction(self,code,offset):
        before,after = self.state_at.get(offset),self.state_after.get(offset)
        before = before if before is not None and before['action']!='DICT_GET' else None
        if before is None and after is None:
            return self.monitor.DISABLE
        frame = sys._getframe(1)
        if after is not None:
            self._commit(frame,after,'NORMAL_NEXT_INSTRUCTION')
        if before is None or self.state_counts[offset]>=self.state_limit or frame not in self.frames:
            return
        try:
            root = _instance_field(self.frames[frame]['receiver'],before['field'])
            if before['action']=='SCALAR_STORE':
                _require(_integer(root),'expected original scalar counter')
                size = None
            else:
                _require(type(root) is dict,'expected exact original dictionary root')
                size = len(root)
            self.pending[frame,offset] = (self.tick(),root,size)
        except (TypeError,ValueError,KeyError):
            self.frontiers['ORIGINAL_STATE_STORE_OPEN'] += 1

    def _commit(self,frame,state,completion):
        selected = state['at']['offset']
        pending = self.pending.pop((frame,selected),None)
        if pending is None or self.state_counts[selected]>=self.state_limit:
            return
        start,root,size = pending
        owner = self.frames[frame]
        receiver = owner['receiver']
        try:
            current = _instance_field(receiver,state['field'])
            if size is None:
                _require(_integer(current) and current==root+1,'original counter increment differs')
                change = dict(before=root,after=current)
            else:
                _require(current is root,'original dictionary root changed over selected operation')
                change = dict(resource_occurrence=self.token(self.roots,root),size_before=size,size_after=len(root))
            self.state_events.append(dict(field=state['field'],action=state['action'],site=state['at'],completion=completion,
                start=start,finish=self.tick(),activation=owner['id'],thread_id=get_ident(),
                receiver_occurrence=self.token(self.receivers,receiver),**change))
            self.state_counts[selected] += 1
        except (TypeError,ValueError,KeyError):
            self.frontiers['ORIGINAL_NORMAL_STATE_COMPLETION_OPEN'] += 1

    def finish(self,code,_offset,result):
        frame = sys._getframe(1)
        owner = self.frames.pop(frame,None)
        try:
            if owner is not None:
                pending = owner['pending']
                probe = self.root_pending.pop(frame,None)
                if pending is not None and self.counts[pending['site_index']]<self.limit:
                    _require(_integer(result),'original recursive return is outside exact scalar subset')
                    pending.update(child_return=result,child_return_sequence=self.tick())
                    pending['state']['return_counter'] = _instance_field(owner['receiver'],self.counter)
                    self.rows.append(pending)
                    self.counts[pending['site_index']] += 1
                if probe is not None:
                    _require(_integer(result),'root probe return is outside exact scalar subset')
                    probe.update(result=result,return_sequence=self.tick(),
                        after=self._root_snapshot(owner['receiver'],owner['entry'][self.position]))
                    self.root_probes.append(probe)
                self.ready.pop(frame,None)
            else:
                phase = self.phases.pop(frame,None)
                if phase is not None:
                    parent,expected,details = phase
                    self.ready[parent] = (expected,details,result,self.tick())
                nested = self.nested.pop(frame,None)
                if nested is not None and nested in self.phases:
                    _require(_integer(result),'nested scorer escaped exact scalar subset')
                    self.phases[nested][2].update(score_observed=True,score_result=result,score_site=self.design['score_call'])
        except (TypeError,ValueError,KeyError):
            self.frontiers['ORIGINAL_NORMAL_RETURN_OPEN'] += 1

    def unwind(self,code,_offset,_exception):
        frame = sys._getframe(1)
        selected = self.frames.pop(frame,None)
        if selected is not None:
            if selected['pending'] is not None:
                self.frontiers['SAMPLED_ORIGINAL_CHILD_UNWOUND'] += 1
            self.ready.pop(frame,None)
            self.root_pending.pop(frame,None)
        self.phases.pop(frame,None)
        self.nested.pop(frame,None)

    def close(self):
        self.active = False
        self.monitor.set_events(self.slot,0)
        for code in self.roles:
            self.monitor.set_local_events(self.slot,code,0)


def capture(design,out,limit=6,state_limit=8):
    reference = _ref(design,DESIGN)
    plan = read_artifact(reference)
    read_artifact(plan['source'],json_document=False)
    _require(type(limit) is int and 1<=limit<=24 and type(state_limit) is int and 1<=state_limit<=32,
             'recursive capture requires bounded observation budgets')
    watch = _Observe(plan,limit,state_limit)
    monitor,events,slot = watch.monitor,watch.events,watch.slot
    callbacks = ((events.PY_START,watch.start),(events.PY_RETURN,watch.finish),(events.PY_UNWIND,watch.unwind),
                 (events.INSTRUCTION,watch.instruction),(events.CALL,watch.call),(events.C_RETURN,watch.c_return))
    monitor.use_tool_id(slot,CAPTURE)
    try:
        for event,callback in callbacks:
            monitor.register_callback(slot,event,callback)
        monitor.set_events(slot,events.PY_UNWIND)
        sys.addaudithook(watch.attach)
        launch = run_original_script(plan['source']['uri'],[])
    finally:
        watch.close()
        for event,_ in callbacks:
            monitor.register_callback(slot,event,None)
        monitor.free_tool_id(slot)
    return _write(out,dict(schema=CAPTURE,schema_version=1,design=reference,source=plan['source'],launch=launch,
        process_id=os.getpid(),samples=watch.rows,state_events=watch.state_events,frontiers=dict(watch.frontiers),
        root_probes=watch.root_probes,module_i64=watch.module_i64,owner_global_reads=watch.global_reads,
        original_execution=True,performance_measurement=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=6)
    args = parser.parse_args()
    capture(args.design,args.out,args.limit)


if __name__ == '__main__':
    main()

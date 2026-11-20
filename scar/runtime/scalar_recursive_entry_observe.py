"""Capture only original startup writes and prepared command-to-search inputs."""

import argparse
import builtins
from collections import Counter
from functools import partial
import locale
import math
import os
from pathlib import Path
import sys
from threading import get_ident
from types import CodeType, FunctionType, ModuleType

from scar.analysis.scalar_recursive_entry import CAPTURE, DESIGN, _require
from scar.analysis.scalar_table_handoff import _ref, _write, table_version
from scar.ir.execution_recipe import read_artifact
from scar.runtime.entry_runner import run_original_script
from scar.runtime.scalar_recursive_driver_observe import ordered_map_digest
from scar.runtime.scalar_recursive_observe import _pod, _receiver_storage
from scar.runtime.source_value_observe import _codes, _coordinate, _target_coordinate
from scar.trace.code_snapshot import describe_code, site

_PRINT = builtins.print


class _Entry:
    def __init__(self,plan):
        self.plan = plan
        self.ui = read_artifact(plan['ui_contract'])
        self.source_driver = read_artifact(plan['source_driver'])
        self.monitor,self.events = sys.monitoring,sys.monitoring.events
        self.slot = next(i for i in range(5,-1,-1) if self.monitor.get_tool(i) is None)
        self.active = True
        self.frame = None
        self.descriptors = {_target_coordinate(row):(role,row) for role,row in (
            ('controller',plan['controller_code']),('consumer',plan['output_protocol']['consumer']),
            ('entry',plan['entry']['body']))}
        self.descriptors.update({_target_coordinate(row['code']):(name,row['code']) for name,row in plan['position_helpers'].items()})
        self.codes = {}
        self.tokens,self.static_prints,self.searches,self.positions = [],[],[],[]
        self.pending = {}
        self.frontiers = Counter()
        self.initial_owner,self.initial_table,self.actual_entry = None,None,None
        self.helpers = {}
        self.print_sites = {row['offset']:row for command in plan['commands'] if command['role']=='STATIC_REPLY'
                            for row in command['print_sites']}
        self.config = list(self.source_driver['sessions'][0]['native_configuration'])

    def attach(self,event,args):
        if (not self.active or event!='exec' or not args or type(args[0]) is not CodeType or args[0].co_filename not in
                (self.plan['source']['uri'],self.plan['ui_source']['uri'])):
            return
        for code in _codes(args[0]):
            selected = self.descriptors.get(_coordinate(code))
            if selected is None:
                continue
            role,descriptor = selected
            _require(describe_code(code)==descriptor,'original entry or helper code identity changed')
            self.codes[code] = role
            mask = self.events.PY_START|self.events.PY_RETURN
            if role=='controller':
                mask |= self.events.INSTRUCTION|self.events.CALL
            if role in self.plan['position_helpers']:
                mask |= self.events.LINE
            self.monitor.set_local_events(self.slot,code,mask)

    def actor(self,frame):
        return frame.f_locals[self.plan['controller']['receiver']]

    def namespace(self,frame):
        module = frame.f_locals[self.plan['controller_code']['parameters'][0]]
        _require(type(module) is ModuleType,'original controller did not receive its exact engine module')
        return vars(module)

    def table(self,frame):
        namespace = self.namespace(frame)
        symbol = self.source_driver['initialization']['table']['symbol']
        table = namespace[symbol]
        _require(type(table) is dict and all(type(key) is str and type(value) is tuple and
                 all(type(item) is int for item in value) for key,value in table.items()),'original table is outside exact scalar rows')
        return table_version({key:list(value) for key,value in table.items()})

    def holder(self,frame):
        storage = _receiver_storage(self.actor(frame))
        return {role:dict(size=len(storage[field]),digest=ordered_map_digest(storage[field]))
                for role,field in self.plan['owner_fields'].items() if type(storage[field]) is dict}

    def startup(self,frame):
        storage = _receiver_storage(self.actor(frame))
        values = {}
        for field,value in storage.items():
            values[field] = value if type(value) is int else _pod(value)
        self.initial_owner = dict(fields=values,holder=self.holder(frame),configuration={name:self.namespace(frame)[name]
                                  for name in self.config})
        self.initial_table = self.table(frame)

    def start(self,code,offset):
        frame = sys._getframe(1)
        parent = frame.f_back
        role = self.codes[code]
        if role=='controller':
            self.frame = frame
            _require(parent is not None and site(parent.f_code,parent.f_lasti)==self.plan['entry']['call'],
                     'original script did not enter its experienced controller call')
            self.actual_entry = dict(entry=site(code,offset),source_call=site(parent.f_code,parent.f_lasti),
                                    thread_id=get_ident())
        elif role=='consumer' and self.searches:
            current = self.searches[-1]
            local = frame.f_locals
            _require('consumer' not in current and local[self.ui['receiver']] is self.actor(self.frame),
                     'original worker did not consume the prepared receiver')
            scalars = {name:value for name in self.ui['consumer']['parameters'] if (value:=local[name]) is None
                       or type(value) in (int,bool) or type(value) is float and math.isfinite(value)}
            _require([_pod(value) for value in local[self.ui['history']]]==current['history'],
                     'worker received a different original prepared history')
            current['consumer'] = dict(code=describe_code(code),entry=site(code,offset),scalars=scalars,thread_id=get_ident())
        elif role in self.plan['position_helpers'] and parent is self.frame:
            descriptor = self.plan['position_helpers'][role]
            _require(site(parent.f_code,parent.f_lasti)==descriptor['caller'],'original position helper has another source caller')
            self.helpers[frame] = dict(name=role,code=descriptor['code'],command_ordinal=len(self.tokens)-1,
                arguments=[_pod(frame.f_locals[name]) for name in descriptor['code']['parameters']],lines=[])

    def line(self,code,number):
        selected = self.helpers.get(sys._getframe(1))
        if selected is not None:
            if number not in selected['lines']:
                selected['lines'].append(number)

    def finish(self,code,offset,value):
        selected = self.helpers.pop(sys._getframe(1),None)
        if selected is not None:
            selected.update(result=_pod(value),return_site=site(code,offset))
            self.positions[-1]['helpers'].append(selected)

    def instruction(self,code,offset):
        control = self.plan['controller']
        is_input = offset==control['command']['after']['offset']
        is_submit = offset==control['submit']['offset']
        if not (is_input or is_submit):
            return self.monitor.DISABLE
        frame = sys._getframe(1)
        if is_input:
            local = frame.f_locals
            tokens = local[control['command']['symbol']]
            ordinal = len(self.tokens)
            _require(ordinal<len(self.plan['commands']) and tokens==self.plan['commands'][ordinal]['tokens'],
                     'running original controller got another command tape')
            self.tokens.append(list(tokens))
            if self.initial_owner is None:
                self.startup(frame)
            if self.plan['commands'][ordinal]['role']=='POSITION':
                self.positions.append(dict(ordinal=ordinal,before=self.holder(frame),table_version=self.table(frame),helpers=[]))
            return
        ordinal = len(self.tokens)-1
        _require(self.plan['commands'][ordinal]['role']=='SEARCH' and self.positions[-1]['ordinal']==ordinal-1,
                 'original submission is outside its prepared position and exact tape')
        handoff = frame.f_locals[control['callable_local']]
        _require(type(handoff) is FunctionType and describe_code(handoff.__code__)==self.ui['consumer'],
                 'original executor selected an unbound consumer')
        history = frame.f_locals[self.ui['history']]
        _require(type(history) is list and 0<len(history)<=256,'prepared original history is not finite exact list')
        holder = self.holder(frame)
        period = self.positions[-1]
        _require(holder==period['before'],'position preparation mutated retained observable holder data')
        position_version = self.table(frame)
        _require(position_version==period['table_version'],'position operation changed the original scoring table')
        period['after'] = holder
        move = holder['move_table']
        self.searches.append(dict(command_ordinal=ordinal,position_ordinal=period['ordinal'],history=[_pod(row) for row in history],
            position_table_version=position_version,submit=site(code,offset),configuration={name:self.namespace(frame)[name]
            for name in self.config},move_table_size=move['size'],move_table_digest=move['digest'],
            halfmove_clock=frame.f_locals[self.plan['halfmove_clock']]))

    def call(self,code,offset,target,arg0):
        event = self.print_sites.get(offset)
        if event is None:
            return self.monitor.DISABLE
        ordinal = len(self.tokens)-1
        frame = sys._getframe(1)
        expected = self.plan['commands'][ordinal]
        if expected['role']!='STATIC_REPLY' or event not in expected['print_sites']:
            return
        _require(type(target) is partial and target.func is _PRINT and not target.args and target.keywords=={'flush':True}
            and type(arg0) is str and arg0.isascii(),'bounded startup output did not use original flush print provider')
        self.pending[frame,offset] = dict(command_ordinal=ordinal,site=event,flush=True,text=arg0)

    def output(self,code,offset,_target,_arg0):
        selected = self.pending.pop((sys._getframe(1),offset),None)
        if selected is not None:
            self.static_prints.append(dict(**selected,normal_return=True))

    def close(self):
        self.active = False
        for code in self.codes:
            self.monitor.set_local_events(self.slot,code,0)


def capture(plan_path,out):
    ref = _ref(plan_path,DESIGN)
    plan = read_artifact(ref)
    for name in ('source','ui_source','stdin'):
        read_artifact(plan[name],json_document=False)
    watch = _Entry(plan)
    monitor,events,slot = watch.monitor,watch.events,watch.slot
    callbacks = ((events.PY_START,watch.start),(events.PY_RETURN,watch.finish),(events.INSTRUCTION,watch.instruction),
                 (events.CALL,watch.call),(events.C_RETURN,watch.output),(events.LINE,watch.line))
    monitor.use_tool_id(slot,CAPTURE)
    try:
        for event,fn in callbacks:
            monitor.register_callback(slot,event,fn)
        sys.addaudithook(watch.attach)
        run = run_original_script(plan['source']['uri'],[])
    finally:
        watch.close()
        for event,_ in callbacks:
            monitor.register_callback(slot,event,None)
        monitor.free_tool_id(slot)
    io = dict(stdin_encoding=sys.stdin.encoding,stdout_encoding=sys.stdout.encoding,stdout_errors=sys.stdout.errors,
              stdout_isatty=sys.stdout.isatty(),locale=locale.setlocale(locale.LC_CTYPE),
              environ={name:os.environ.get(name) for name in ('LANG','LC_ALL','PYTHONHASHSEED')})
    return _write(out,dict(schema=CAPTURE,schema_version=1,design=ref,source=plan['source'],stdin=plan['stdin'],
        input_tokens=watch.tokens,positions=watch.positions,initial_owner=watch.initial_owner,
        initial_table=watch.initial_table,actual_entry=watch.actual_entry,searches=watch.searches,
        static_prints=watch.static_prints,io=io,frontiers=dict(watch.frontiers),launch=run,
        original_execution=True,performance_measurement=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    capture(args.design,args.out)


if __name__=='__main__':
    main()

"""Record the original recursive driver's finite probe/yield transcript."""

import argparse
from collections import Counter
from dis import get_instructions
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from threading import get_ident
from types import CodeType

from scar.analysis.scalar_recursive_driver import CAPTURE, SCHEMA, _require
from scar.analysis.scalar_table_handoff import _ref, _write, table_version
from scar.ir.execution_recipe import read_artifact
from scar.runtime.entry_runner import run_original_script
from scar.runtime.scalar_recursive_observe import _pod, _receiver_storage
from scar.runtime.source_value_observe import _codes, _coordinate, _target_coordinate
from scar.trace.code_snapshot import describe_code, site


def ordered_map_digest(value):
    """Hash typed entries in insertion order without retaining another map."""
    digest=sha256()
    for key,item in value.items():
        row=json.dumps([_pod(key),_pod(item)],ensure_ascii=True,sort_keys=True,separators=(',',':'))
        digest.update(row.encode('ascii')+b'\n')
    return digest.hexdigest()


class _Driver:
    def __init__(self,plan,configuration_ports):
        self.plan = plan
        self.monitor,self.events = sys.monitoring,sys.monitoring.events
        self.slot = next(index for index in range(5,-1,-1) if self.monitor.get_tool(index) is None)
        self.active = True
        self.code,self.frames,self.probes = {},{},{}
        self.sessions,self.objects,self.frontiers = [],[],Counter()
        self.sequence = 0
        self.configuration_ports = configuration_ports
        self.receivers = {role:plan['initialization'][role]['field'] for role in ('counter','history','score_table','root')}
        self.receivers['move_table'] = plan['yield']['move_lookup']['field']
        self.wanted = {_target_coordinate(desc):(role,desc) for role,desc in
            (('driver',plan['driver']['code']),('probe',plan['probe']['source_recursive_owner']))}

    def attach(self,event,args):
        if (not self.active or event!='exec' or not args or type(args[0]) is not CodeType
                or args[0].co_filename!=self.plan['source']['uri']):
            return
        for code in _codes(args[0]):
            selected = self.wanted.get(_coordinate(code))
            if selected is None:
                continue
            role,descriptor = selected
            _require(describe_code(code)==descriptor,'original driver or probe identity differs')
            self.code[role] = code
            mask = self.events.PY_START|self.events.PY_RETURN
            if role=='driver':
                mask |= self.events.PY_YIELD|self.events.PY_RESUME
            self.monitor.set_local_events(self.slot,code,mask)

    def tick(self):
        self.sequence += 1
        return self.sequence

    def identity(self,obj):
        for token,other in enumerate(self.objects,1):
            if obj is other:
                return token
        _require(len(self.objects)<32,'driver identity observation budget exhausted')
        self.objects.append(obj)
        return len(self.objects)

    def _snapshot(self,receiver):
        storage = _receiver_storage(receiver)
        table = {}
        for role in ('history','score_table','move_table'):
            value = storage[self.receivers[role]]
            _require(type(value) is (set if role=='history' else dict),'driver storage protocol differs')
            table[role] = dict(identity=self.identity(value),size=len(value))
        nodes = storage[self.receivers['counter']]
        _require(type(nodes) is int,'expected exact original driver counter')
        return dict(**table,nodes=nodes,root=_pod(storage.get(self.receivers['root'])))

    def _table_version(self,frame):
        record = self.plan['initialization']['table']
        namespace = frame.f_globals
        root = namespace[record['symbol']]
        _require(type(root) is dict and all(type(key) is str and type(row) is tuple and
            all(type(value) is int for value in row) for key,row in root.items()) and len(root)<=32,
            'driver score rows do not have an exact small dictionary layout')
        digest = table_version({key:list(row) for key,row in root.items()})
        _require(digest in record['versions'],'driver selected a new unadmitted score table version')
        actual = root[record['key']]
        alternatives = [key for key in ('true_row','false_row') if actual is namespace[record[key]]]
        _require(len(alternatives)==1,'selected original score row does not identify one source alternative')
        return digest,self.identity(root),alternatives[0]

    def start(self,code,offset):
        frame = sys._getframe(1)
        parent = frame.f_back
        if code is self.code['driver']:
            if len(self.sessions)+len(self.frames)>=4:
                return
            params = self.plan['driver']
            loc = frame.f_locals
            receiver = loc[params['receiver']]
            try:
                history = loc[params['history_formal']]
                _require(type(history) in (list,tuple) and 0<len(history)<=256,'expected original finite history')
                names = {op.argval for role in ('driver','probe') for op in get_instructions(self.code[role]) if op.opname=='LOAD_GLOBAL'}
                globals_ = {name:frame.f_globals[name] for name in sorted(names)
                            if name in frame.f_globals and type(frame.f_globals[name]) is int}
                ancestors = []
                for _ in range(2):
                    if parent is None:
                        break
                    ancestors.append(dict(code=describe_code(parent.f_code),site=site(parent.f_code,parent.f_lasti)))
                    if len(ancestors)==2:
                        consumer_locals=parent.f_locals
                    parent = parent.f_back
                data = dict(identity=self.identity(receiver),thread_id=get_ident(),begin_sequence=self.tick(),
                    entry=site(code,offset),caller_chain=ancestors,history_input_size=len(history),
                    root_input=_pod(history[-1]),initial=self._snapshot(receiver),module_i64=globals_,yields=[])
                if self.configuration_ports:
                    _require(len(ancestors)==2,'bounded study requires an outer original consumer')
                    data['native_configuration']={name:frame.f_globals[name] for name in self.configuration_ports}
                    _require(all(type(value) is int for value in data['native_configuration'].values()),
                             'native configuration ports require integers')
                    data['consumer_scalars']={name:value for name in ancestors[-1]['code']['parameters']
                        if (value:=consumer_locals[name]) is None or type(value) in (int,bool)
                        or type(value) is float and math.isfinite(value)}
                    data['history_input']=[_pod(value) for value in history]
                    data['initial_move_digest']=ordered_map_digest(_receiver_storage(receiver)[self.receivers['move_table']])
                self.frames[frame] = dict(receiver=receiver,data=data,probe=None,thrown=None)
            except (KeyError,ValueError,TypeError):
                self.frontiers['DRIVER_ENTRY_STATE_OPEN'] += 1
            return
        if parent not in self.frames or parent.f_lasti!=self.plan['probe']['call']['offset']:
            return
        state = self.frames[parent]
        try:
            _require(site(parent.f_code,parent.f_lasti)==self.plan['probe']['call'],'original recursive probe site changed')
            values = frame.f_locals
            receiver,pos,gamma,depth,root = code.co_varnames[:5]
            _require(values[receiver] is state['receiver'] and values[pos] is
                parent.f_locals[self.plan['initialization']['root']['local']] and values[root] is True
                and type(values[gamma]) is type(values[depth]) is int,'recursive root call did not bind its original actuals')
            data = state['data']
            if 'initialized' not in data:
                digest,root_token,alternative = self._table_version(parent)
                data.update(initialized=self._snapshot(state['receiver']),table_version=digest,
                            table_identity=root_token,table_alternative=alternative)
            self.probes[frame] = (parent,dict(actuals=[values[gamma],values[depth],True],
                sequence=self.tick(),same_receiver=True,same_root_position=True))
        except (KeyError,ValueError,TypeError):
            self.frontiers['DRIVER_ROOT_PROBE_OPEN'] += 1

    def yielded(self,code,offset,value):
        frame = sys._getframe(1)
        state = self.frames.get(frame)
        if state is None:
            return
        try:
            rows = state['data']['yields']
            p = state['probe']
            state['probe'] = None
            _require(len(rows)<128 and p is not None and type(value) is tuple and len(value)==4
                and site(code,offset)==self.plan['yield']['site'],'original driver yield did not link one bounded returned probe')
            depth,gamma,score,move = value
            carrier = None if move is None else _pod(move)
            if type(carrier) is dict:
                carrier = carrier['fields']
            snap = self._snapshot(state['receiver'])
            loc = frame.f_locals
            names = self.plan['bracket']
            rows.append(dict(value=[depth,gamma,score,carrier],probe=p,sequence=self.tick(),site=site(code,offset),
                lower=loc[names['lower']],upper=loc[names['upper']],nodes=snap['nodes'],
                resources={role:snap[role] for role in ('score_table','move_table')},resume=None))
        except (KeyError,ValueError,TypeError):
            self.frontiers['DRIVER_YIELD_OPEN'] += 1

    def resumed(self,code,offset):
        frame = sys._getframe(1)
        state = self.frames.get(frame)
        if state and state['data']['yields']:
            state['data']['yields'][-1]['resume'] = site(code,offset)

    def finish(self,code,offset,value):
        frame = sys._getframe(1)
        probe = self.probes.pop(frame,None)
        if probe is not None:
            parent,data = probe
            if parent in self.frames:
                data.update(result=value,return_sequence=self.tick(),return_site=site(code,offset))
                self.frames[parent]['probe'] = data
        if frame in self.frames:
            self._complete(frame,dict(kind='NORMAL_RETURN',site=site(code,offset)))

    def thrown(self,code,offset,exception):
        if code is self.code.get('driver'):
            state = self.frames.get(sys._getframe(1))
            if state is not None:
                state['thrown'] = dict(site=site(code,offset),type=type(exception).__qualname__)

    def unwind(self,code,offset,exception):
        if code is self.code.get('driver'):
            frame = sys._getframe(1)
            if frame in self.frames:
                self._complete(frame,dict(kind='GENERATOR_CLOSE' if type(exception) is GeneratorExit else 'EXCEPTION_UNWIND',
                    site=site(code,offset),exception=type(exception).__qualname__,thrown=self.frames[frame]['thrown']))
        elif code is self.code.get('probe') and self.probes.pop(sys._getframe(1),None) is not None:
            self.frontiers['DRIVER_ROOT_PROBE_UNWOUND'] += 1

    def _complete(self,frame,completion):
        state = self.frames.pop(frame)
        data = state['data']
        if data['yields']:
            data.update(final=self._snapshot(state['receiver']),exit=dict(**completion,sequence=self.tick()))
            if self.configuration_ports:
                storage=_receiver_storage(state['receiver'])
                data['final']['ordered_map_digests']={field:ordered_map_digest(storage[field])
                    for field in (self.receivers['move_table'],self.receivers['score_table'])}
            self.sessions.append(data)

    def close(self):
        self.active = False
        self.monitor.set_events(self.slot,0)
        for code in self.code.values():
            self.monitor.set_local_events(self.slot,code,0)


def capture(plan_path,out,configuration_ports=()):
    ref = _ref(plan_path,SCHEMA)
    plan = read_artifact(ref)
    read_artifact(plan['source'],json_document=False)
    _require(plan['capture'] is None,'capture must use a static original source design')
    study = _Driver(plan,configuration_ports)
    monitor,events,slot = study.monitor,study.events,study.slot
    callbacks = ((events.PY_START,study.start),(events.PY_YIELD,study.yielded),(events.PY_RESUME,study.resumed),
                 (events.PY_RETURN,study.finish),(events.PY_THROW,study.thrown),(events.PY_UNWIND,study.unwind))
    monitor.use_tool_id(slot,CAPTURE)
    try:
        for event,callback in callbacks:
            monitor.register_callback(slot,event,callback)
        monitor.set_events(slot,events.PY_THROW|events.PY_UNWIND)
        sys.addaudithook(study.attach)
        execution = run_original_script(plan['source']['uri'],[])
    finally:
        study.close()
        for event,_ in callbacks:
            monitor.register_callback(slot,event,None)
        monitor.free_tool_id(slot)
    return _write(out,dict(schema=CAPTURE,schema_version=1,design=ref,source=plan['source'],launch=execution,
        sessions=study.sessions,frontiers=dict(study.frontiers),process_id=os.getpid(),
        original_execution=True,performance_measurement=False))


def capture_stream(design, stdin, protocol_plan, cpu, out, configuration_ports=()):
    """Bind the original diagnostic, raw command input and its process launch."""
    from scar.compiler.cpu_entry_replay import _pace, _paced_transcript
    from scar.optimization.scalar_recursive_parameterized_plan import SCHEMA as PROTOCOL, _load_plan, _stdout

    if type(cpu) is not int or cpu not in os.sched_getaffinity(0):
        raise ValueError('captured stream requires one available CPU')
    design = Path(design).resolve(strict=True)
    design_ref = _ref(design, SCHEMA)
    stdin = _ref(Path(stdin).resolve(strict=True), 'application/octet-stream')
    protocol = _ref(Path(protocol_plan).resolve(strict=True), PROTOCOL)
    plan = _load_plan(protocol['uri'])
    if plan['scope']['source'] != read_artifact(design_ref)['source']:
        raise ValueError('pacing plan identifies a different original source')
    payload = read_artifact(stdin, json_document=False)
    transcript, effects, depths = _paced_transcript(payload, plan)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    raw, stdout, stderr = (out/name for name in ('raw.json', 'stdout.bin', 'stderr.bin'))
    command = ['/usr/bin/taskset', '-c', str(cpu), '/usr/bin/timeout', '-k', '2', '120',
               str(Path(sys.executable).resolve()), '-B', '-u', '-m',
               'scar.runtime.scalar_recursive_driver_observe',
               '--design', str(design), '--out', str(raw)]
    for port in configuration_ports:
        command.extend(('--configuration-port', port))
    environment = dict(PATH=os.defpath, LANG='C.UTF-8', LC_ALL='C', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0')
    cwd = str(Path.cwd().resolve())
    with stdout.open('xb') as output, stderr.open('xb') as errors:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                              env=environment, cwd=cwd) as process:
            events = _pace(process, output, transcript, record=True)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
    if stderr.stat().st_size or read_artifact(stdin, json_document=False) != payload:
        raise ValueError('stream capture changed its input or emitted diagnostic stderr')
    launch = out/'launch.json'
    launch.write_text(json.dumps(dict(command=command, cwd=cwd, environment=environment, stdin=stdin['uri'],
        protocol_plan=protocol, pacing='WAIT_FOR_MATCHED_STDOUT_PREFIX',
        stdin_mode='PACED_PIPE', pacing_events=events,
        normalized_stdout_sha256=_stdout(stdout.read_bytes(), effects)),
                                 indent=2, sort_keys=True)+'\n')
    raw_ref = _ref(raw, CAPTURE)
    observation = read_artifact(raw_ref)
    if (observation['design'] != design_ref or observation['frontiers'] or observation['launch']['exit_code'] != 0
            or len(observation['sessions']) != len(depths)):
        raise ValueError('stream capture did not observe a complete original diagnostic')
    read_artifact(design_ref, json_document=False)
    read_artifact(protocol, json_document=False)
    return _write(out/'result.json', dict(observation, raw_capture=raw_ref, stdin=stdin,
        stdout=_ref(stdout, 'application/octet-stream'), captured_launch=_ref(launch, 'application/octet-stream')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--configuration-port',action='append',default=[])
    parser.add_argument('--stdin',type=Path,help='capture a sealed original input stream into an output directory')
    parser.add_argument('--cpu',type=int,help='exact CPU for a sealed stream capture')
    parser.add_argument('--protocol-plan',type=Path,help='original typed command and acknowledgement plan')
    args = parser.parse_args()
    specified = (args.stdin, args.cpu, args.protocol_plan)
    if any(value is not None for value in specified) and any(value is None for value in specified):
        parser.error('--stdin, --cpu and --protocol-plan must be provided together')
    if args.stdin is None:
        capture(args.design,args.out,args.configuration_port)
    else:
        print(capture_stream(args.design,args.stdin,args.protocol_plan,args.cpu,args.out,args.configuration_port)['uri'])


if __name__ == '__main__':
    main()

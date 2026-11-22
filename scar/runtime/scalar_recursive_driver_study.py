"""Connect an admitted resident native driver to its original fixed-depth consumer."""

import argparse
import ast
from collections import Counter
import ctypes as ffi
from importlib import import_module
import json
import math
from pathlib import Path
import struct
import sys
from time import perf_counter_ns, time as wall_clock
from types import FunctionType

from scar.analysis import scalar_score_reuse_evidence as source
from scar.codegen.scalar_recursive_leaves import render as render_leaves
from scar.codegen.scalar_recursive_lowering import _state_shape
from scar.codegen.scalar_recursive_standalone import _encoded
from scar.codegen.scalar_sorted_library import pod_types
from scar.compiler.scalar_table_region import _definition
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.scalar_score_ir import _int64, _require
from scar.runtime.csr_region_entry import _observer_active
from scar.runtime.entry_runner import run_original_script
from scar.runtime.scalar_recursive_driver_observe import ordered_map_digest
from scar.runtime.scalar_recursive_observe import _pod
from scar.runtime.scalar_sorted_study import _namespace_exact
from scar.runtime.scalar_table_contract import MISS, OFFER
from scar.runtime.scalar_table_contract import _run_main
from scar.trace.code_snapshot import describe_code


SCHEMA = 'scar.scalar-recursive-driver-source-study'
UI = 'scar.scalar-recursive-ui-contract'
FRONTEND = 'scar.scalar-recursive-driver'
LIBRARY = 'scar.scalar-recursive-driver-library'
_LOCAL = '_scar_recursive_driver_session_v1'
_READER = ffi.CFUNCTYPE(ffi.c_double, ffi.c_void_p)
_DONE, _STOP = 1, 4


def _name(node):
    return node.id if type(node) is ast.Name else None


def _short(descriptor):
    return descriptor['qualname'].rsplit('.',1)[-1]


def project_ui(ui, driver_path, request_port, limit_port, constraints):
    driver_ref = artifact_reference(Path(driver_path).resolve(strict=True), FRONTEND)
    driver = read_artifact(driver_ref)
    ui_ref = artifact_reference(Path(ui).resolve(strict=True), 'scar.python-source')
    module, codes = source._compile(ui_ref)
    method = _short(driver['driver']['code'])
    choices=[]
    for function in module.body:
        if type(function) is not ast.FunctionDef:
            continue
        for loop in ast.walk(function):
            if type(loop) is not ast.For or type(loop.iter) is not ast.Call:
                continue
            relay = loop.iter
            calls = [arg for arg in relay.args if type(arg) is ast.Call and type(arg.func) is ast.Attribute
                     and arg.func.attr==method and len(arg.args)==1 and not arg.keywords]
            if len(calls)==1 and _name(relay.func):
                target=calls[0]
                choices.append((function,relay.func.id,_name(target.func.value),_name(target.args[0])))
    consumer,relay_name,receiver,history=source._one(choices,'expected one original driver consumer and relay')
    relay=_definition(module,relay_name)
    _require(receiver and history and any(type(node) is ast.YieldFrom for node in ast.walk(relay)),
             'consumer must pass the original owned generator through its relay')
    params = {node.arg for node in consumer.args.posonlyargs+consumer.args.args+consumer.args.kwonlyargs}
    _require({receiver,history,request_port,limit_port,*constraints}<=params and
             all(type(value) in (int,bool,type(None)) for value in constraints.values()),
             'declared fixed-depth mode differs from the original consumer signature')
    consumer_code,relay_code=describe_code(codes[consumer.name]),describe_code(codes[relay_name])
    _require(len({receiver,history,request_port,limit_port,*constraints})==4+len(constraints)
             and constraints and driver['sessions'], 'fixed-depth ports and observed sessions are required')
    for session in driver['sessions']:
        mode=session['consumer_scalars']
        request=mode.get(request_port)
        _require([row['code'] for row in session['caller_chain']]==[relay_code,consumer_code]
                 and type(request) is int and request>0
                 and type(mode.get(limit_port)) is int and mode[limit_port]==request
                 and session['exit']['kind']=='GENERATOR_CLOSE'
                 and session['yields'][-1]['value'][0]==request+1
                 and set(constraints)<=set(mode)
                 and all(type(mode[name]) is type(value) and mode[name]==value
                         for name,value in constraints.items()),'consumer mode differs from original observed scope')
    return dict(schema=UI,schema_version=1,ui_source=ui_ref,source_driver=driver_ref,
                consumer=consumer_code,relay=relay_code,
                receiver=receiver,history=history,request_port=request_port,limit_port=limit_port,
                constraints=constraints,native_scope='FIXED_DEPTH_ORIGINAL_CONSUMER_WITHOUT_OBSERVERS',
                live_foreign_aliases_proven_absent=False,deployment_authorized=False)


def _seal(driver):
    raw=read_artifact(driver['source'],json_document=False)
    module,codes=source._compile(driver['source'])
    desc=driver['driver']['code']
    function=_definition(module,desc['qualname'])
    original=codes[desc['qualname']]
    reserved={_LOCAL,OFFER,MISS}
    _require(describe_code(original)==desc and all(not reserved.intersection((*code.co_names,*code.co_varnames))
                                                  for code in codes.values()),'driver source or reserved bindings differ')
    gamma=driver['initialization']['gamma']['store']['position']
    index=source._one((i for i,node in enumerate(function.body) if type(node) is ast.Assign
                      and any(source._span(target)==gamma for target in node.targets)),'cannot locate original driver start')
    invocation=ast.Call(func=ast.Name(id=OFFER,ctx=ast.Load()),
                        args=[ast.Name(id=driver['driver'][role],ctx=ast.Load())
                              for role in ('receiver','history_formal')],keywords=[])
    assign=ast.NamedExpr(target=ast.Name(id=_LOCAL,ctx=ast.Store()),value=invocation)
    test=ast.Compare(left=assign,ops=[ast.IsNot()],comparators=[ast.Constant(value=None)])
    delegated=ast.Expr(value=ast.YieldFrom(value=ast.Name(id=_LOCAL,ctx=ast.Load())))
    branch=ast.If(test=test,body=[delegated,ast.Return(value=None)],orelse=[])
    function.body.insert(index,ast.copy_location(branch,function.body[index]))
    recipe=dict(filename=desc['filename'],mode='exec',flags=0,dont_inherit=True,optimize=0)
    code=compile(ast.fix_missing_locations(module),recipe['filename'],'exec',dont_inherit=True,optimize=0)
    from scar.compiler.scalar_table_region import code_tree

    revised=source._one((item for item in code_tree(code) if item.co_qualname==desc['qualname']
                        and item.co_firstlineno==desc['first_line']),'revised driver owner differs')
    shape=('co_argcount','co_posonlyargcount','co_kwonlyargcount','co_flags','co_freevars','co_cellvars')
    _require(all(getattr(revised,key)==getattr(original,key) for key in shape),'driver rewrite changed its original ABI')
    return raw,recipe,revised


def _clock_value(value):
    _require(type(value) in (int,float) and math.isfinite(value)
             and float(value)==value,'clock deadline is outside the exact finite scalar domain')
    return float(value)


def _tuple_constructor(cls, fields):
    from collections import namedtuple

    _require(type(cls) is type and issubclass(cls,tuple) and cls._fields==tuple(fields)
             and cls.__eq__ is tuple.__eq__ and cls.__hash__ is tuple.__hash__
             and cls.__getitem__ is tuple.__getitem__ and cls.__iter__ is tuple.__iter__
             and cls.__getattribute__ is tuple.__getattribute__ and cls.__init__ is tuple.__init__
             and getattr(cls,'__del__',None) is None,'source record tuple shape or observation effects changed')
    constructor=cls.__new__
    prototype=namedtuple('_RecursiveDriverRecord',fields).__new__
    _require(type(constructor) is FunctionType and constructor.__code__==prototype.__code__
             and constructor.__defaults__ is None and constructor.__kwdefaults__ is None
             and constructor.__globals__.get('_tuple_new') is tuple.__new__,'source tuple allocation changed')
    from _collections import _tuplegetter

    for i,name in enumerate(fields):
        slot=getattr(cls,name)
        _require(type(slot) is _tuplegetter and slot.__reduce__()[1][0]==i,'source tuple descriptor changed')


class _Codec:
    def __init__(self, namespace, sort_plan, record_type, state_class, entry_class, entry_fields):
        self.namespace=namespace
        self.position=state_class
        self.move=namespace[sort_plan['constructor']]
        self.entry=namespace[entry_class]
        self.state_fields=[sort_plan['state_fields'][name] for name in sort_plan['state_field_order']]
        self.move_fields=sort_plan['record_types']
        self.record_type=record_type
        _tuple_constructor(self.position,sort_plan['state_field_order'])
        _tuple_constructor(self.move,sort_plan['record_fields'])
        _tuple_constructor(self.entry,entry_fields)

    @staticmethod
    def _scalar(value, kind):
        _require((kind=='bool' and type(value) is bool) or (kind=='int' and _int64(value))
                 or (kind=='str' and type(value) is str and value.isascii() and '\0' not in value),
                 'original field violates the admitted POD contract')
        return value

    def flatten(self, value, cls, fields):
        _require(type(value) is cls and tuple.__len__(value)==len(fields),'original state or record carrier differs')
        result=[]
        for datum,kind in zip(tuple.__iter__(value),fields,strict=True):
            if kind=='bool_pair':
                _require(type(datum) is tuple and len(datum)==2,'expected the original boolean-pair carrier')
                result.extend(self._scalar(item,'bool') for item in datum)
            else:
                result.append(self._scalar(datum,kind))
        return tuple(result)

    def position_bytes(self, position):
        return _encoded(self.flatten(position,self.position,self.state_fields)).encode('ascii')

    def move_record(self, row):
        values=[]
        for port in self.record_type:
            kind,slot=port['type'],port['slot']
            if kind=='str':
                size=row.lengths[slot]
                offset=port['offset']
                _require(size<=port['capacity'],'native yielded invalid ASCII length')
                value=bytes(row.strings[offset:offset+size]).decode('ascii')
            else:
                value=row.scalars[slot]
            values.append(value)
        return tuple.__new__(self.move,values)

    def pack_world(self, storage, kinds):
        blocks=[]
        for name,slot in kinds.items():
            value=storage[name]
            kind=slot['type']
            if kind=='float':
                blocks.append(_clock_value(value))
            elif kind=='int':
                blocks.append(self._scalar(value,'int'))
            elif kind=='state':
                blocks.append(self.flatten(value,self.position,self.state_fields))
            elif kind=='set':
                _require(type(value) is set,'original history requires exact set storage')
                blocks.extend((len(value),[self.flatten(item,self.position,self.state_fields) for item in value]))
            elif kind in ('move_map','entry_map'):
                _require(type(value) is dict,'original ordered state requires exact dict storage')
                rows=[]
                for key,entry in value.items():
                    if kind=='entry_map':
                        _require(type(key) is tuple and len(key)==2,'sampled ordered key protocol changed')
                        key_shape=(self.flatten(key[0],self.position,self.state_fields),self._scalar(key[1],'int'))
                        item=self.flatten(entry,self.entry,('int','int'))
                    else:
                        key_shape=self.flatten(key,self.position,self.state_fields)
                        item=self.flatten(entry,self.move,self.move_fields)
                    rows.append((key_shape,item))
                blocks.extend((len(rows),rows))
            else:
                raise ValueError('unexpected driver state field type')
        return _encoded(blocks).encode('ascii')

    def decode_world(self, raw, kinds):
        rows=iter(raw.split(b'\n'))
        _require(next(rows)==b'SCAR_DRIVER_WORLD_V1','native driver snapshot does not match its wire')
        def scalar(kind):
            value=next(rows)
            if kind=='str':
                return bytes.fromhex(value.decode('ascii')).decode('ascii')
            number=int(value)
            if kind=='bool':
                _require(number in (0,1),'invalid native boolean state')
                return bool(number)
            if kind=='float':
                return struct.unpack('=d',struct.pack('=Q',number))[0]
            return number

        def record(cls, fields):
            leaves=[(scalar('bool'),scalar('bool')) if kind=='bool_pair' else scalar(kind) for kind in fields]
            return tuple.__new__(cls,leaves)

        def position():
            return record(self.position,self.state_fields)

        result={}
        for name,slot in kinds.items():
            kind=slot['type']
            if kind=='state':
                value=position()
            elif kind=='set':
                value={position() for _ in range(scalar('int'))}
            elif kind=='move_map':
                value={position():record(self.move,self.move_fields) for _ in range(scalar('int'))}
            elif kind=='entry_map':
                value={(position(),scalar('int')):record(self.entry,('int','int')) for _ in range(scalar('int'))}
            else:
                value=scalar(kind)
            result[name]=value
        _require(next(rows,None)==b'' and next(rows,None) is None,'unexpected data follows native world snapshot')
        return result


class _Library:
    def __init__(self, path, sort_abi):
        lib=ffi.CDLL(str(path))
        _,self.Record=pod_types(sort_abi)
        class Row(ffi.Structure):
            _fields_=[('depth',ffi.c_int64),('threshold',ffi.c_int64),('score',ffi.c_int64),('nodes',ffi.c_int64),
                      ('has_move',ffi.c_uint32),('table_slot',ffi.c_uint32),('move',self.Record)]
        self.Row=Row
        ptr,size,opaque=ffi.c_void_p,ffi.c_size_t,ffi.c_char_p
        signatures={
            'create':((opaque,size,ffi.POINTER(ptr)),ffi.c_int),
            'advance':((ptr,ffi.c_double,ffi.c_double,ffi.POINTER(Row)),ffi.c_int),
            'get_move':((ptr,opaque,size,ffi.POINTER(self.Record)),ffi.c_int),
            'snapshot':((ptr,ptr,size,ffi.POINTER(size)),ffi.c_int),
            'close':((ptr,),None),
        }
        optional=getattr(lib,'scar_recursive_driver_set_deadline_reader_v1',None)
        self.deadline_reader=None
        if optional is not None:
            optional.argtypes=(ptr,_READER,ptr)
            optional.restype=ffi.c_int
            self.deadline_reader=optional
        for name,(args,result) in signatures.items():
            symbol=getattr(lib,'scar_recursive_driver_'+name+'_v1')
            symbol.argtypes,symbol.restype=args,result
            setattr(self,name,symbol)


class _Moves:
    """The original consumer performs read-only lookups against the current native state."""

    def __init__(self, offer, handle, codec):
        self.offer,self.handle,self.codec=offer,handle,codec
        self.row=offer.library.Record()

    def get(self, position, default=None):
        started=self.offer.tick()
        data=self.codec.position_bytes(position)
        code=self.offer.library.get_move(self.handle,data,len(data),ffi.byref(self.row))
        _require(code in (0,_DONE),'native move lookup fell outside the declared consumer protocol')
        value=default if code==_DONE else self.codec.move_record(self.row)
        self.offer.count('lookups')
        self.offer.elapsed('lookup',started)
        return value


class _Offer:
    def __init__(self, driver, ui, recursive, sort_bundle, sort, guard, native, binary, revised, *, observe):
        self.driver,self.ui,self.recursive,self.sort,self.guard,self.native=driver,ui,recursive,sort,guard,native
        self.library=_Library(binary,sort_bundle['abi'])
        self.revised=revised
        self.observe=observe
        self.live=0
        self.completed=0
        self.statistics=Counter()
        self.timings=Counter()
        self.sessions=[]
        self.frontiers=Counter()
        self.states=recursive['world']
        self.config=recursive['configuration_ports']
        self.table_slots=native['table_slots']
        self.counter=driver['initialization']['counter']['field']
        self.root=driver['initialization']['root']['field']
        self.history=driver['initialization']['history']['field']
        self.score_map=driver['initialization']['score_table']['field']
        self.move_map=driver['yield']['move_lookup']['field']
        self.soft=driver['time_and_exit']['soft']['deadline_field']
        self.deadline=source._one((name for name,value in self.states.items() if value['type']=='float'),
                                  'compiled region has no unique clock deadline port')
        owner=driver['probe']['source_recursive_owner']
        module,codes=source._compile(driver['source'])
        function=_definition(module,owner['qualname'])
        _, entries, clocks=_state_shape(module,function,owner['parameters'][0],owner['parameters'][1])
        self.entry=source._one(entries,'recursive state pair has no original class')
        self.entry_fields=entries[self.entry]
        self.clocks=clocks
        self.recursion_owner=owner
        self.exception=source._one((node.exc.id for node in ast.walk(function)
                                   if type(node) is ast.Raise and type(node.exc) is ast.Name),
                                   'original checked stop type is missing')
        state_plan=read_artifact(sort['transition'])
        transition=read_artifact(state_plan['evidence'])
        leaves=render_leaves(sort,state_plan)
        self.roles=[sort['factory'],leaves['witness']['code'],
                    *(transition[name]['code'] for name in ('callee','rotation','score'))]
        self.expected={label:{key:tuple(row) for key,row in entry['rows'].items()}
                       for label,entry in state_plan['table_versions'].items()}

    def tick(self):
        return perf_counter_ns() if self.observe else 0

    def count(self,key,amount=1):
        if self.observe:
            self.statistics[key]+=amount

    def elapsed(self,name,start):
        if self.observe:
            self.timings[name]+=perf_counter_ns()-start

    def _ui_mode(self,current,actor,history,expected):
        relay=current.f_back
        consumer=relay.f_back if relay is not None else None
        if consumer is None or describe_code(relay.f_code)!=self.ui['relay'] or describe_code(consumer.f_code)!=self.ui['consumer']:
            return False
        local=consumer.f_locals
        return (local.get(self.ui['receiver']) is actor and local.get(self.ui['history']) is history
                and all(name in local and type(local[name]) is type(value) and local[name]==value
                        for name,value in expected['consumer_scalars'].items()))

    def _guard(self, current, owner, history):
        _require(not _observer_active() and not self.live and self.completed<len(self.driver['sessions']),
                 'no matching observed native session remains')
        expected=self.driver['sessions'][self.completed]
        _require(self._ui_mode(current,owner,history,expected),
                 'original call is outside the fixed-depth consumer study')
        namespace,builtin=current.f_globals,current.f_builtins
        actor=type(owner)
        source_class=self.recursion_owner['qualname'].split('.')[0]
        _require(_namespace_exact(namespace) and _namespace_exact(builtin) and type(actor) is type
                 and namespace.get(source_class) is actor
                 and actor.__getattribute__ is object.__getattribute__ and actor.__setattr__ is object.__setattr__,
                 'recursive receiver or namespace binding changed')
        descriptors=[self.recursion_owner,*self.roles]
        for desc in descriptors:
            class_name,_,name=desc['qualname'].rpartition('.')
            cls=namespace.get(class_name)
            method=vars(cls).get(name) if type(cls) is type else None
            _require(type(method) is FunctionType and method.__globals__ is namespace
                     and method.__builtins__ is builtin and method.__closure__ is None
                     and describe_code(method.__code__)==desc,'an original native leaf dispatch changed')
        state_cls=namespace[self.sort['factory']['qualname'].split('.')[0]]
        codec=_Codec(namespace,self.sort,self.native['abi']['move']['ports'][1:],
                     state_cls,self.entry,self.entry_fields)
        # Primitive operators in the admitted detached scalar region must resolve
        # to their ordinary builtins or imported providers at this source boundary.
        import builtins
        for name in ('abs','all','any','int','iter','len','max','min','next','range','set','sorted','enumerate'):
            _require(namespace.get(name,builtin.get(name)) is getattr(builtins,name),
                     'an original exact builtin has changed')
        for name,(module,export) in self.sort['builtin_calls'].items():
            _require(namespace.get(name) is getattr(import_module(module),export),'native iterator provider changed')
        clock=import_module('time')
        _require(clock.time is wall_clock and all(namespace.get(name) is clock for name in self.clocks),
                 'recursive clock provider differs from the original')
        constants=self.guard['bindings']['immutable_i64']
        _require(all(type(namespace.get(name)) is int and namespace[name]==value for name,value in constants.items()),
                 'immutable original case constants changed')
        for name,map_ in self.sort['integer_rows'].items():
            actual=namespace.get(name)
            _require(_namespace_exact(actual) and all(type(row) is tuple and all(_int64(item) for item in row)
                    for row in actual.values()) and actual=={key:tuple(row) for key,row in map_.items()},
                     'a source iterator map changed')
        table=namespace.get(self.driver['initialization']['table']['symbol'])
        _require(_namespace_exact(table) and all(type(row) is tuple and all(_int64(item) for item in row)
                 for row in table.values()),'original scoring table type changed')
        label=source._one((version for version,expected in self.expected.items() if expected==table),
                          'current source scoring table does not match either frozen version')
        storage=vars(owner)
        _require(type(history) is list and history and type(storage) is dict and storage[self.root] is history[-1]
                 and type(storage[self.score_map]) is dict and not storage[self.score_map]
                 and type(storage[self.move_map]) is dict and self.recursion_owner['qualname'].rsplit('.',1)[-1] not in storage
                 and storage[self.counter]==self.driver['initialization']['counter']['initial'],
                 'original retained initialization or owner storage differs')
        config=[codec._scalar(namespace[name],'int') for name in self.config]
        _require(dict(zip(self.config,config,strict=True))==expected['native_configuration']
                 and label==expected['table_version'] and len(history)==expected['history_input_size']
                 and [_pod(value) for value in history]==expected['history_input']
                 and len(storage[self.move_map])==expected['initial']['move_table']['size']
                 and ordered_map_digest(storage[self.move_map])==expected['initial_move_digest'],
                 'live configuration, input or retained ordered state falls outside the observed study')
        seed=(b'SCAR_RECURSIVE_INPUT_V1\n'+label.encode()+b'\n'+codec.position_bytes(storage[self.root])
              +_encoded((0,0,True)).encode()+codec.pack_world(storage,self.states)+_encoded(config).encode())
        return namespace,codec,storage,seed,self.table_slots.index(label)

    def __call__(self, actor, history):
        started=self.tick()
        frame=sys._getframe(1)
        try:
            _require(frame.f_code is self.revised,'unrecognized driver call boundary')
            namespace,codec,storage,seed,slot=self._guard(frame,actor,history)
            soft,deadline=_clock_value(storage[self.soft]),_clock_value(storage[self.deadline])
        except (ValueError,TypeError,KeyError,AttributeError,OverflowError) as error:
            self.count('fallback')
            if self.observe:
                self.frontiers[str(error)]+=1
            return None
        handle=ffi.c_void_p()
        result=self.library.create(seed,len(seed),ffi.byref(handle))
        if result!=0:
            self.count('fallback')
            if self.observe:
                self.frontiers[f'DRIVER_CREATE_STATUS_{result}']+=1
            return None
        self.live+=1
        self.count('native_sessions')
        self.elapsed('guard_and_begin',started)
        return self._drive(handle,codec,storage,namespace[self.exception],slot,soft,deadline)

    def _snapshot(self,handle,codec):
        size=ffi.c_size_t()
        status=self.library.snapshot(handle,None,0,ffi.byref(size))
        _require(status==1 and size.value>0,'native world did not report its snapshot capacity')
        raw=ffi.create_string_buffer(size.value)
        status=self.library.snapshot(handle,raw,len(raw),ffi.byref(size))
        _require(status==0 and size.value==len(raw),'native owner did not return its full state')
        self.count('snapshot_bytes',size.value)
        return codec.decode_world(raw.raw,self.states)

    def _drive(self, handle, codec, storage, stop, slot, soft, deadline):
        old_moves,old_scores=storage[self.move_map],storage[self.score_map]
        holder=None
        publish=False
        trace=None
        try:
            holder=_Moves(self,handle,codec)
            row=self.library.Row()
            failure=[]
            if self.observe:
                trace=dict(table_version=self.table_slots[slot],initial_move_entries=len(old_moves),
                           initial_score_entries=len(old_scores),yields=[],outcome='consumer_close')
                self.sessions.append(trace)
            if self.library.deadline_reader is not None:
                def live_deadline(context):
                    try:
                        self.count('deadline_reads')
                        return _clock_value(storage[self.deadline])
                    except (TypeError,ValueError,OverflowError) as error:
                        failure.append(error)
                        return 0.0
                reader=_READER(live_deadline)
                _require(self.library.deadline_reader(handle,reader,None)==0,'native scope rejected the deadline provider')
            storage[self.move_map]=holder
            while True:
                publish=False
                started=self.tick()
                status=self.library.advance(handle,soft,deadline,ffi.byref(row))
                self.elapsed('native_advance',started)
                self.count('native_advances')
                _require(not failure,'foreign deadline left the admitted domain')
                if status==_DONE:
                    publish=True
                    self.count('native_exhausted')
                    if trace is not None:
                        trace['outcome']='source_exhausted'
                    break
                if status==_STOP:
                    publish=True
                    self.count('native_stop')
                    if trace is not None:
                        trace['outcome']='source_stop'
                    raise stop
                _require(status==0 and row.table_slot==slot and row.has_move in (0,1),
                         'native driver left its admitted region')
                storage[self.counter]=row.nodes
                move=codec.move_record(row.move) if row.has_move else None
                self.count('native_yields')
                if trace is not None:
                    trace['yields'].append(dict(value=[row.depth,row.threshold,row.score,
                                                     list(move) if move is not None else None],nodes=row.nodes))
                publish=True
                yield row.depth,row.threshold,row.score,move
                soft,deadline=_clock_value(storage[self.soft]),_clock_value(storage[self.deadline])
        finally:
            try:
                if publish:
                    started=self.tick()
                    _require(storage.get(self.move_map) is holder and storage.get(self.score_map) is old_scores,
                             'an external writer intervened in the native owned state')
                    values=self._snapshot(handle,codec)
                    for field, original in ((self.move_map,old_moves),(self.score_map,old_scores)):
                        original.clear()
                        original.update(values[field])
                        storage[field]=original
                    storage[self.counter]=values[self.counter]
                    self.count('native_closed')
                    self.elapsed('snapshot_and_publish',started)
                    if trace is not None:
                        trace.update(final_move_entries=len(old_moves),final_score_entries=len(old_scores),
                                     final_nodes=values[self.counter],ordered_map_digests={
                                         self.move_map:ordered_map_digest(old_moves),
                                         self.score_map:ordered_map_digest(old_scores)})
                    self.completed+=1
            finally:
                if holder is not None and storage.get(self.move_map) is holder:
                    storage[self.move_map]=old_moves
                self.library.close(handle)
                self.live-=1


def _pinned(path,schema,digest):
    ref=artifact_reference(Path(path).resolve(strict=True),schema)
    _require(ref['sha256']==digest,'source trial input did not match its pinned digest')
    return ref,read_artifact(ref)


def run(entry,argv,args):
    library_ref,native=_pinned(args.library,LIBRARY,args.sha256)
    driver_ref,driver=_pinned(args.source_driver,FRONTEND,args.driver_sha256)
    ui_ref,ui=_pinned(args.ui_contract,UI,args.ui_sha256)
    _require(ui['source_driver']==driver_ref and str(entry)==driver['source']['uri'],
             'UCI consumer and native driver must share the same original executable source')
    _require(ui==project_ui(ui['ui_source']['uri'],driver_ref['uri'],ui['request_port'],ui['limit_port'],ui['constraints']),
             'fixed-depth consumer contract differs from its original source')
    base=Path(library_ref['uri']).parent
    binary=(base/native['library']['uri']).resolve(strict=True)
    _require(binary.is_relative_to(base),'driver DSO must remain a local package member')
    read_artifact({**native['library'],'uri':str(binary)},json_document=False)
    recursive=read_artifact(native['native'])
    fusion=read_artifact(recursive['fusion'])
    sort_bundle=read_artifact(fusion['sorted'])
    sort=read_artifact(sort_bundle['region'])
    region=read_artifact(driver['recursive_region'])
    flow=read_artifact(region['original_fusion_flow'])
    guard=read_artifact(flow['source_guard'])
    _require(native['code']==driver['driver']['code'] and recursive['region']==driver['recursive_region']
             and guard['plan']==sort_bundle['region'] and native['world']==recursive['world']
             and native['world_order']==list(recursive['world'])
             and native['configuration_ports']==recursive['configuration_ports']
             and native['table_slots']==sort_bundle['abi']['table_slots']
             and native['abi']['move']==sort_bundle['abi']['record']
             and native['world'].get(native['clock_fields']['deadline'],{}).get('type')=='float'
             and native['clock_fields']['soft']==driver['time_and_exit']['soft']['deadline_field'],
             'native scope must resolve the exact original graph chain')
    raw,recipe,revised=_seal(driver)
    offer=_Offer(driver,ui,recursive,sort_bundle,sort,guard,native,binary,revised,observe=args.observe)
    execution=_run_main(entry,argv,dict(compile_recipe=recipe,original_code_key=driver['driver']['code']['key']),
                        raw,revised,offer)
    _require(offer.live==0,'a native handle outlived the original driver frame')
    return dict(schema=SCHEMA,schema_version=1,source=driver['source'],library=library_ref,source_driver=driver_ref,
                ui_contract=ui_ref,execution=execution,statistics=dict(offer.statistics) if args.observe else None,
                phase_ns=dict(offer.timings) if args.observe else None,
                sessions=offer.sessions if args.observe else None,frontiers=dict(offer.frontiers) if args.observe else None,
                admitted_for_deployment=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path)
    parser.add_argument('--sha256')
    parser.add_argument('--source-driver',type=Path)
    parser.add_argument('--driver-sha256')
    parser.add_argument('--ui-contract',type=Path)
    parser.add_argument('--ui-sha256')
    parser.add_argument('--project-ui',type=Path)
    parser.add_argument('--request-port')
    parser.add_argument('--limit-port')
    parser.add_argument('--constraints',type=json.loads)
    parser.add_argument('--isolated-study',action='store_true')
    parser.add_argument('--observe',action='store_true')
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv)
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    native=(args.library,args.sha256,args.source_driver,args.driver_sha256,args.ui_contract,args.ui_sha256)
    if args.project_ui is not None:
        if (command or args.isolated_study or args.observe or any(native[:2]+native[3:])
                or not all((args.source_driver,args.request_port,args.limit_port,args.constraints))):
            parser.error('project a consumer with --source-driver, --request-port, --limit-port and --constraints')
        report=project_ui(args.project_ui,args.source_driver,args.request_port,args.limit_port,args.constraints)
    elif (not command or args.isolated_study and not all(native) or
            any((args.request_port,args.limit_port,args.constraints)) or
            not args.isolated_study and (any(native) or args.observe)):
        parser.error('provide an original script, or pinned library, driver and consumer with --isolated-study')
    else:
        entry=Path(command[0]).resolve(strict=True)
        report=(run(entry,command[1:],args) if args.isolated_study else
                dict(schema=SCHEMA,schema_version=1,source=None,library=None,source_driver=None,ui_contract=None,
                     execution=run_original_script(entry,command[1:]),statistics=None,phase_ns=None,
                     admitted_for_deployment=False))
        report.update(bridge=artifact_reference(Path(__file__).resolve(),'scar.python-source'),
                      python_version=sys.version)
    with args.report.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')


if __name__=='__main__':
    main()

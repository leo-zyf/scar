"""Compact ABI loading and live guards for an admitted ordinary array caller.

No compiler, graph, planner or code generator is imported here. Native use
remains an explicit candidate experiment; unknown cost always defaults to the
original entire caller. The unchanged host withholds private outputs on error.
"""
from contextlib import contextmanager
from copy import deepcopy
import ctypes
from dataclasses import dataclass,field
from functools import wraps
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import platform
import re
import struct
import sys
import threading
from types import FunctionType,ModuleType

from .native_loop import (_Array,_descriptor,_cpu_available,_file_hash_and_header,
    _small_raw,_json,GuardFailure)
from .stateful_guard import _function,_function_state,_same_function_state


SCHEMA='scar.caller-pipeline-runtime-contract'
ADMISSION='scar.caller-pipeline-runtime-admission'
TRUST='EXPLICIT_EXPECTED_ADMISSION_SHA256_FROM_BUILD_PRODUCER_NOT_A_SIGNATURE'
_RANGE=range
_GETTRACE=sys.gettrace;_GETPROFILE=sys.getprofile
_MONITOR=getattr(sys,'monitoring',None)
_MONITOR_APIS=None if _MONITOR is None else (_MONITOR.get_tool,_MONITOR.get_events,_MONITOR.get_local_events)
_PARALLEL_FAMILY='DEPENDENCY_WAVES_CPU_SERIAL_LEAVES_V1'


def _need(condition,detail):
    if not condition:raise ValueError(detail)


def _hash(value):return type(value)is str and re.fullmatch('[0-9a-f]{64}',value)is not None
def _basename(value):return type(value)is str and value not in('','.','..')and Path(value).name==value
def _source(value):
    _need(type(value)is dict and set(value)=={'path','sha256','code_key','qualname'}
        and type(value['path'])is str and Path(value['path']).is_absolute()and _hash(value['sha256'])
        and type(value['code_key'])is str and re.fullmatch('pycode-v2:[0-9a-f]{64}',value['code_key'])is not None
        and type(value['qualname'])is str and bool(value['qualname']),'original compact source/CodeID required')


def implementation_sha256():
    root=Path(__file__).resolve().parents[1]
    names=('runtime/caller_pipeline_guard.py','runtime/native_loop.py','runtime/stateful_guard.py',
        'runtime/array_copy_guard.py','trace/code_snapshot.py')
    rows=[(name,sha256((root/name).read_bytes()).hexdigest())for name in names]
    return sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()


def validate_contract(value):
    _need(type(value)is dict and value.get('schema')==SCHEMA and value.get('schema_version')==1,
        'supported caller runtime contract required')
    _source(value['source']);parameters=value['parameters'];inputs=value['inputs']
    _need(type(parameters)is list and 0<len(parameters)<=4096 and len(set(parameters))==len(parameters)
        and all(type(name)is str and name.isidentifier()for name in parameters)
        and type(inputs)is dict and set(inputs)==set(parameters),'ordered original caller parameters required')
    for row in (*inputs.values(),value['output']):
        shape=row['shape']
        _need(type(shape)is list and len(shape)==2 and all(type(n)is int and 0<=n<=sys.maxsize//8 for n in shape)
            and type(row['nbytes'])is int and row['nbytes']==shape[0]*shape[1]*8 and row['nbytes']<=sys.maxsize
            and row['dtype']=='float64'and row['layout']=='C_CONTIGUOUS','exact bounded array specialization required')
    callees=value['callees'];names=set()
    _need(type(callees)is list and 0<len(callees)<=4096,'bounded original global callees required')
    for row in callees:
        _source(row['source']);name=row['global_name']
        _need(type(name)is str and name.isidentifier()and name not in names,'unique direct global callee required')
        names.add(name)
    native=value['native'];operators=native['operators']
    _need(native['host_abi_version']==native['operator_abi_version']==1
        and _hash(native['pipeline_sha256'])and _hash(native['memory_plan_sha256'])
        and type(operators)is list and 2<=len(operators)<=4096,'original pooled host/operator ABI1 required')
    for row in (native['host'],*operators):
        _need(_basename(row['library'])and _hash(row['sha256'])and type(row['bytes'])is int and row['bytes']>=64,
            'original library bytes/receipt required')
    for row in operators:
        _need(_hash(row['plan_sha256'])and type(row['input_count'])is int and 0<row['input_count']<=4096,
            'original operator input/plan identity required')
    cpu=value['cpu'];ids=cpu['ids']
    _need(type(ids)is list and ids and len(set(ids))==len(ids)and all(type(i)is int and 0<=i<2**20 for i in ids)
        and cpu['os']=='linux'and cpu['architecture']=='x86_64'and cpu['pointer_bits']==64
        and cpu['byteorder']=='little'and cpu['required_isa']=='BASELINE_X86_64','original supported CPU binding required')
    parallel=('execution_family'in native,'execution_schedule_sha256'in native,'execution_schedule_receipt'in native)
    _need(all(parallel)or not any(parallel),'complete optional dependency-wave control contract required')
    if all(parallel):
        receipt=native['execution_schedule_receipt']
        _need(native['execution_family']==_PARALLEL_FAMILY and _hash(native['execution_schedule_sha256'])
            and type(receipt)is dict and set(receipt)=={'path','sha256','bytes'}
            and type(receipt['path'])is str and Path(receipt['path']).is_absolute()
            and _hash(receipt['sha256'])and type(receipt['bytes'])is int and receipt['bytes']>0,
            'original dependency-wave schedule identity/producer receipt required')
        _need(all(row.get('cpu_threads')==1 and type(row.get('cpu_threads'))is int
            and type(row.get('cpu_ids'))is list and len(row['cpu_ids'])==1
            and row['cpu_ids'][0]in ids for row in operators),
            'dependency-wave operators require their exact single-CPU placement')
    memory=value['memory']
    _need(memory['policy']=='UNBOUND'and memory['scope']=='INCREMENTAL_PAYLOAD_WORKSPACE_CAPACITIES_PLUS_FRESH_FINAL'
        and all(type(memory[key])is int and 0<=memory[key]<=sys.maxsize for key in('workspace_bytes','final_bytes','peak_bytes','budget_bytes'))
        and memory['final_bytes']==value['output']['nbytes']
        and memory['peak_bytes']==memory['workspace_bytes']+memory['final_bytes']<=memory['budget_bytes'],
        'original feasible host payload contract required')
    _need(value['cost_status']=='UNKNOWN'and value['automatic_deployment_authorized']is False
        and value['execution_policy']=='NO_OP_UNLESS_EXPLICIT_CANDIDATE_MEASUREMENT'
        and value['defaults_policy']=='NO_DEFAULT_PARAMETERS'
        and value['input_writes']is False and value['fresh_output']is True,
        'guarded unmeasured caller profile required')
    if 'array_copy_contract' in value:
        from .array_copy_guard import runtime_copy_contract
        _need(value['array_copy_contract']==runtime_copy_contract(),
            'exact ndarray C-order copy runtime contract required')
    return value


_OperatorRun=ctypes.CFUNCTYPE(ctypes.c_int32,ctypes.POINTER(_Array),ctypes.c_uint64,ctypes.POINTER(_Array))
class _OperatorTable(ctypes.Structure):
    _fields_=[('abi_version',ctypes.c_uint32),('struct_size',ctypes.c_uint32),('input_count',ctypes.c_uint64),
        ('run',_OperatorRun),('plan_sha256',ctypes.c_char_p)]
_TablePointer=ctypes.POINTER(_OperatorTable)


@dataclass
class NativeCallerPipeline:
    directory:Path
    _contract:dict
    _report:dict
    _host:object
    _operators:tuple
    _tables:object
    _run:object
    _statistics:dict=field(default_factory=lambda:dict(native_dispatches=0,native_calls=0))

    @property
    def contract(self):return deepcopy(self._contract)
    @property
    def report(self):return {**deepcopy(self._report),**self._statistics}

    def run(self,*arrays):
        contract=self._contract;_cpu_available(contract['cpu']['ids'])
        if len(arrays)!=len(contract['parameters']):raise GuardFailure('caller native array count differs')
        if 'array_copy_contract'in contract:
            from .array_copy_guard import guard_copy_inputs,CopyGuardFailure
            try:guard_copy_inputs(arrays)
            except CopyGuardFailure as error:raise GuardFailure(str(error))from error
        import numpy as np
        descriptors=[]
        for name,array in zip(contract['parameters'],arrays):
            descriptor=_descriptor(array,np)
            if list(array.shape)!=contract['inputs'][name]['shape']or not array.flags.c_contiguous:
                raise GuardFailure('caller shape/layout differs from contract')
            descriptors.append(descriptor)
        try:output=np.empty(contract['output']['shape'],dtype=np.float64)
        except (MemoryError,ValueError,OverflowError)as error:raise GuardFailure('caller final allocation declined')from error
        final=_descriptor(output,np)
        if not output.flags.owndata or any(row.lower_bound<final.upper_bound and final.lower_bound<row.upper_bound for row in descriptors):
            raise GuardFailure('caller output is not a private fresh allocation')
        failed=ctypes.c_uint64(2**64-1);self._statistics['native_dispatches']+=1
        status=self._run((_Array*len(descriptors))(*descriptors),len(descriptors),ctypes.byref(final),self._tables,
            len(contract['native']['operators']),ctypes.byref(failed))
        self._statistics['last_status']=int(status)
        if status:raise GuardFailure('private caller native output declined: '+str(status)+' at '+str(failed.value))
        self._statistics['native_calls']+=1
        return output


def _admit(bundle,expected_admission_sha256):
    _need(_hash(expected_admission_sha256),'explicit expected admission SHA256 required')
    directory=Path(bundle).resolve();raw=_small_raw(directory/'admission.json')
    _need(sha256(raw).hexdigest()==expected_admission_sha256,'caller admission trust pin differs')
    admission=_json(raw)
    _need(admission.get('schema')==ADMISSION and admission.get('schema_version')==1 and admission['trust_policy']==TRUST,
        'supported caller runtime admission required')
    _need(admission['runtime_implementation_sha256']==implementation_sha256(),'caller runtime implementation changed')
    raw=_small_raw(directory/'runtime.json')
    _need(sha256(raw).hexdigest()==admission['runtime_sha256'],'caller runtime contract bytes changed')
    contract=validate_contract(_json(raw));native=contract['native']
    _need(native==admission['native'],'caller native identities differ from admission')
    def verify_library(row):
        path=directory/row['library'];digest,header=_file_hash_and_header(path)
        _need(digest==row['sha256']and path.stat().st_size==row['bytes']and len(header)>=64
            and header[:6]==b'\x7fELF\x02\x01'and struct.unpack_from('<H',header,16)[0]==3
            and struct.unpack_from('<H',header,18)[0]==62,'caller library bytes or ELF64 ABI changed')
    members={'runtime.json','admission.json',native['host']['library'],*(row['library']for row in native['operators'])}
    _need(members<={p.name for p in directory.iterdir()}<=members|{'export-report.json'},'unexpected caller bundle members')
    for row in (native['host'],*native['operators']):verify_library(row)
    return directory,contract,admission


def inspect_bundle(bundle,*,expected_admission_sha256):
    """Validate compact metadata and ELF receipts without loading libraries."""
    directory,contract,admission=_admit(bundle,expected_admission_sha256)
    return dict(directory=str(directory),contract=deepcopy(contract),admission=deepcopy(admission),
        admission_sha256=expected_admission_sha256,metadata_only=True,library_bytes_checked=True,library_loaded=False)


def load(bundle,*,expected_admission_sha256):
    """Load unchanged ELF/ABI tables using only producer-pinned compact bytes."""
    directory,contract,admission=_admit(bundle,expected_admission_sha256);native=contract['native']
    _need(sys.platform=='linux'and platform.machine().lower()in('x86_64','amd64')and sys.byteorder=='little'
        and struct.calcsize('P')==8 and ctypes.sizeof(_Array)==64 and ctypes.sizeof(_OperatorTable)==32,
        'caller host requires the original Linux x86_64 ABI1')
    _cpu_available(contract['cpu']['ids'])
    def library(row):return ctypes.CDLL(str(directory/row['library']))
    operators=[];pointers=[]
    for row in native['operators']:
        binary=library(row);getter=binary.scar_get_operator_table_v1
        getter.argtypes=[];getter.restype=_TablePointer;table=getter()
        _need(bool(table)and table.contents.abi_version==1 and table.contents.struct_size>=ctypes.sizeof(_OperatorTable)
            and table.contents.input_count==row['input_count']and bool(table.contents.run)
            and table.contents.plan_sha256==row['plan_sha256'].encode(),'caller operator table identity/ABI differs')
        operators.append(binary);pointers.append(table)
    host=library(native['host'])
    for symbol,digest in(('scar_pipeline_plan_sha256_v1',native['pipeline_sha256']),
            ('scar_pipeline_memory_sha256_v1',native['memory_plan_sha256'])):
        getter=getattr(host,symbol);getter.argtypes=[];getter.restype=ctypes.c_char_p
        _need(getter()==digest.encode(),'caller host embedded plan/memory differs')
    if native.get('execution_family')==_PARALLEL_FAMILY:
        getter=host.scar_pipeline_schedule_sha256_v1;getter.argtypes=[];getter.restype=ctypes.c_char_p
        _need(getter()==native['execution_schedule_sha256'].encode(),
            'caller host embedded dependency-wave schedule differs')
    run=host.scar_pipeline_run_v1
    run.argtypes=[ctypes.POINTER(_Array),ctypes.c_uint64,ctypes.POINTER(_Array),ctypes.POINTER(_TablePointer),
        ctypes.c_uint64,ctypes.POINTER(ctypes.c_uint64)];run.restype=ctypes.c_int32
    report=dict(admission_sha256=expected_admission_sha256,runtime_sha256=admission['runtime_sha256'],
        pipeline_sha256=native['pipeline_sha256'],memory_plan_sha256=native['memory_plan_sha256'],
        automatic_deployment_authorized=False,cost_status='UNKNOWN',native_bytes_preserved=True)
    return NativeCallerPipeline(directory,contract,report,host,tuple(operators),(_TablePointer*len(pointers))(*pointers),run)


def _plain(function,source):
    _function(source,function)
    if (type(function.__globals__)is not dict or type(function.__builtins__)is not dict
            or function.__closure__ is not None
            or function.__defaults__ is not None or function.__kwdefaults__ is not None
            or '__signature__'in vars(function)or'__wrapped__'in vars(function)):
        raise ValueError('caller requires unchanged plain functions without default/signature/closure state')


def _observers(codes):
    if _GETTRACE()is not None or _GETPROFILE()is not None:return True
    if _MONITOR_APIS is not None:
        get_tool,get_events,get_local=_MONITOR_APIS
        for index in _RANGE(6):
            if get_tool(index)is not None or get_events(index)or any(get_local(index,code)for code in codes):return True
    return False


def guarded_caller_pipeline(original,artifact,*,exclusive_inputs=False,measure_candidate=False,selection=None):
    """Guard current bindings and run native, or call the entire original once.

    A selection dictionary never enables execution. Sourced performance and
    deployment admission are deliberately outside this candidate-only adapter.
    Input contents are recomputed on every native call, never memoized.
    """
    if type(artifact)is not NativeCallerPipeline:raise TypeError('loaded compact caller artifact required')
    contract=validate_contract(artifact.contract)
    if type(original)is not FunctionType:raise ValueError('ordinary original function required')
    stats=dict(native_calls=0,fallback_calls=0,reasons={});binding_reason=None;callees=[];signature=None
    try:
        _plain(original,contract['source']);original_state=_function_state(original)
        shadow=FunctionType(original.__code__,original.__globals__);signature=inspect.signature(shadow,follow_wrapped=False)
        if tuple(signature.parameters)!=tuple(contract['parameters']):raise ValueError('original caller parameter order differs')
        for row in contract['callees']:
            function=original.__globals__.get(row['global_name']);_plain(function,row['source'])
            callees.append((row,function,_function_state(function)))
    except (OSError,ValueError,TypeError,KeyError):binding_reason='initial_source_callable_or_abi_not_admitted'
    def fallback(reason,args,kwargs):
        stats['fallback_calls']+=1;stats['reasons'][reason]=stats['reasons'].get(reason,0)+1
        return original(*args,**kwargs)
    @wraps(original)
    def call(*args,**kwargs):
        if measure_candidate is not True:return fallback('NO_OP_cost_unknown_no_candidate_measurement',args,kwargs)
        if exclusive_inputs is not True or threading.active_count()!=1:
            return fallback('exclusive_synchronous_input_contract_required',args,kwargs)
        if binding_reason:return fallback(binding_reason,args,kwargs)
        try:
            if _observers((original.__code__,*(function.__code__ for _,function,_ in callees))):
                raise ValueError('global or local metadata observer active')
            _plain(original,contract['source'])
            if not _same_function_state(original,original_state):raise ValueError('original caller state changed')
            for row,function,state in callees:
                _plain(function,row['source'])
                if original.__globals__.get(row['global_name'])is not function or not _same_function_state(function,state):
                    raise ValueError('original global callee binding/default/code changed')
            bound=signature.bind(*args,**kwargs);bound.apply_defaults()
            arrays=[bound.arguments[name]for name in contract['parameters']]
            if 'array_copy_contract'in contract:
                from .array_copy_guard import guard_copy_inputs
                guard_copy_inputs(arrays)
            import numpy as np
            for name,value in zip(contract['parameters'],arrays):
                if (type(value)is not np.ndarray or value.dtype!=np.dtype('float64')or not value.dtype.isnative
                        or value.ndim!=2 or list(value.shape)!=contract['inputs'][name]['shape']
                        or not value.flags.c_contiguous or not value.flags.aligned):
                    raise ValueError('input metadata outside original specialization')
            _cpu_available(contract['cpu']['ids'])
        except (OSError,ValueError,TypeError,KeyError,GuardFailure):return fallback('live_source_abi_or_input_guard_declined',args,kwargs)
        try:result=artifact.run(*arrays)
        except GuardFailure:return fallback('native_private_output_guard_declined',args,kwargs)
        stats['native_calls']+=1
        return result
    call.scar_stats=stats;call.scar_artifact=artifact;call.scar_runtime_contract=deepcopy(contract)
    return call


@contextmanager
def install_caller_pipeline(module,artifact,*,exclusive_inputs=False,measure_candidate=False):
    """Temporarily guard one direct module caller without changing its source."""
    name=artifact.contract['source']['qualname']
    if type(module)is not ModuleType or not name.isidentifier():raise ValueError('ordinary direct module caller required')
    original=vars(module)[name];wrapped=guarded_caller_pipeline(original,artifact,
        exclusive_inputs=exclusive_inputs,measure_candidate=measure_candidate)
    setattr(module,name,wrapped)
    try:yield wrapped
    finally:
        if vars(module).get(name)is wrapped:setattr(module,name,original)


__all__=['NativeCallerPipeline','validate_contract','inspect_bundle','load','guarded_caller_pipeline',
    'install_caller_pipeline','implementation_sha256']

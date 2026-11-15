"""Pinned temporal-loop delivery without compiler, graph or planner imports."""
from copy import deepcopy
import ctypes
from dataclasses import dataclass
from functools import wraps
from hashlib import sha256
import inspect
from pathlib import Path
import platform
import re
import struct
import sys
import threading
from types import FunctionType

from .native_loop import (GuardFailure,_Array,_descriptor,_small_raw,_json,
    _file_hash_and_header,_cpu_available)
from .loop_guard import SourceMismatch,_state,_same_state,_range
from scar.trace.code_snapshot import code_key

_RANGE=range
SCHEMA='scar.temporal-invariant-runtime-contract'
ADMISSION='scar.temporal-invariant-runtime-admission'
TRUST='EXPLICIT_EXPECTED_ADMISSION_SHA256_FROM_EXPORT_PRODUCER_NOT_A_SIGNATURE'


def _need(ok,message):
    if not ok:raise ValueError(message)
def _hash(value):return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None
def _source(value):
    _need(type(value) is dict and Path(value['path']).is_absolute() and _hash(value['sha256'])
        and re.fullmatch('pycode-v2:[0-9a-f]{64}',value['code_key']) is not None
        and type(value['qualname']) is str and value['qualname'],'invalid original callable identity')


def validate_contract(value):
    _need(type(value) is dict and value.get('schema')==SCHEMA and value.get('schema_version')==1,
        'versioned temporal runtime contract required')
    _source(value['source']);_source(value['leaf']['source'])
    parameters=value['parameters'];mapping=value['parameter_map'];arrays=value['arrays']
    _need(type(parameters) is list and len(set(parameters))==len(parameters)
        and all(type(n) is str and n.isidentifier() for n in parameters),'original outer parameters required')
    _need(type(mapping) is dict and set(mapping)==set(arrays) and mapping
        and all(v in parameters for v in mapping.values()) and value['carry_parameter'] in mapping
        and value['trip_count_parameter'] in parameters and value['trip_count_parameter'] not in mapping.values(),
        'exact outer/leaf input mapping required')
    _need(type(value['input_order']) is list and len(value['input_order'])==len(mapping)
        and set(value['input_order'])==set(mapping),'exact native input descriptor order required')
    _need(type(value['leaf']['global_name']) is str and value['leaf']['global_name'].isidentifier(),
        'direct global leaf binding required')
    shapes=[]
    for name,spec in arrays.items():
        shape=spec['shape']
        _need(type(name) is str and type(shape) is list and len(shape)==2
            and all(type(n) is int and n>0 for n in shape) and spec['dtype']=='float64'
            and spec['layout']=='C_CONTIGUOUS','exact finite array specialization required')
        shapes.append(shape)
    _need(all(shape==shapes[0] for shape in shapes),'temporal input shapes must match')
    count=value['trip_count'];budget=value['memory_budget_bytes']
    _need(type(count) is int and 0<count<2**63 and type(budget) is int and budget>=8
        and shapes[0][0]*shapes[0][1]*8<=budget<=sys.maxsize
        and shapes[0][0]*shapes[0][1]*count<2**64,'bounded positive temporal specialization required')
    _need(value['cpu']['os']=='linux' and value['cpu']['architecture']=='x86_64'
        and value['cpu']['pointer_bits']==64 and value['cpu']['byteorder']=='little'
        and value['cpu']['required_isa']=='BASELINE_X86_64'
        and type(value['cpu']['ids']) is list and len(value['cpu']['ids'])==1
        and all(type(c) is int and 0<=c<2**20 for c in value['cpu']['ids']),
        'supported single-CPU execution boundary required')
    native=value['native']
    _need(Path(native['library']).name==native['library'] and native['library'] not in
        ('runtime.json','admission.json','export-report.json') and _hash(native['library_sha256'])
        and native['operator_descriptor_abi_version']==native['temporal_control_abi_version']==1,
        'admitted temporal descriptor/control ABI required')
    _need(value['profitability_status']=='UNPROVEN_NO_SOURCED_TIMING_MODEL'
        and value['deployment_authorized'] is False,'unmeasured export cannot authorize deployment')
    return value


@dataclass
class NativeTemporalInvariant:
    directory: Path
    _contract: dict
    _report: dict
    _library: object
    _run: object

    @property
    def contract(self):return deepcopy(self._contract)
    @property
    def report(self):return deepcopy(self._report)

    def run(self,inputs,count):
        contract=self._contract
        if type(count) is not int or count!=contract['trip_count']:
            raise GuardFailure('temporal loop count differs from exact specialization')
        if type(inputs) is not dict or set(inputs)!=set(contract['arrays']):
            raise GuardFailure('temporal leaf input mapping differs')
        _cpu_available(contract['cpu']['ids'])
        import numpy as np
        descriptors=[]
        for name in contract['input_order']:
            value=inputs[name];descriptor=_descriptor(value,np)
            if list(value.shape)!=contract['arrays'][name]['shape'] or not value.flags.c_contiguous:
                raise GuardFailure('temporal shape/layout differs from scheduled specialization')
            descriptors.append(descriptor)
        shape=contract['arrays'][contract['carry_parameter']]['shape']
        try:output=np.empty(shape,dtype=np.float64)
        except (MemoryError,ValueError,OverflowError) as error:
            raise GuardFailure('temporal fresh final allocation declined') from error
        descriptor=_descriptor(output,np);counters=(ctypes.c_uint64*2)()
        status=self._run((_Array*len(descriptors))(*descriptors),len(descriptors),count,ctypes.byref(descriptor),counters)
        self._report['last_execution']=dict(native_calls=1,element_preheaders=counters[0],
            element_iterations=counters[1],logical_iterations=count,status=status)
        if status:raise GuardFailure('temporal private native output declined: '+str(status))
        return output


def load(bundle,*,expected_admission_sha256):
    _need(_hash(expected_admission_sha256),'explicit expected export admission SHA256 required')
    directory=Path(bundle).resolve();raw=_small_raw(directory/'admission.json')
    _need(sha256(raw).hexdigest()==expected_admission_sha256,'temporal admission trust pin differs')
    admission=_json(raw)
    _need(admission.get('schema')==ADMISSION and admission.get('schema_version')==1
        and admission['trust_policy']==TRUST,'supported temporal export admission required')
    raw=_small_raw(directory/'runtime.json')
    _need(sha256(raw).hexdigest()==admission['runtime_sha256'],'temporal runtime contract changed')
    contract=validate_contract(_json(raw));native=contract['native']
    _need(native['library_sha256']==admission['library_sha256'],'temporal library admission differs')
    _need(sys.platform=='linux' and platform.machine().lower() in ('x86_64','amd64')
        and sys.byteorder=='little' and struct.calcsize('P')==8 and ctypes.sizeof(_Array)==64,
        'temporal runtime requires Linux x86_64 descriptor ABI')
    _cpu_available(contract['cpu']['ids'])
    members={p.name for p in directory.iterdir()}
    _need({'admission.json','runtime.json',native['library']}<=members<=
        {'admission.json','runtime.json',native['library'],'export-report.json'},'unexpected temporal bundle members')
    digest,header=_file_hash_and_header(directory/native['library'])
    _need(digest==native['library_sha256'] and len(header)>=64 and header[:6]==b'\x7fELF\x02\x01'
        and struct.unpack_from('<H',header,16)[0]==3 and struct.unpack_from('<H',header,18)[0]==62,
        'temporal shared library bytes or ELF ABI differ')
    library=ctypes.CDLL(str(directory/native['library']));run=library.scar_temporal_invariant_loop_v1
    run.argtypes=[ctypes.POINTER(_Array),ctypes.c_uint64,ctypes.c_uint64,
        ctypes.POINTER(_Array),ctypes.POINTER(ctypes.c_uint64)];run.restype=ctypes.c_int32
    return NativeTemporalInvariant(directory,contract,dict(admission_sha256=expected_admission_sha256,
        library_sha256=digest,last_execution=None,profitability_status=contract['profitability_status']),library,run)


def guarded_temporal_invariant(original,artifact,*,exclusive_inputs=False,trial=False,selection=None):
    contract=validate_contract(artifact.contract);leafspec=contract['leaf']
    if type(original) is not FunctionType or original.__closure__ or code_key(original.__code__)!=contract['source']['code_key']:
        raise SourceMismatch('original temporal loop callable differs from export')
    leaf=original.__globals__.get(leafspec['global_name'])
    if type(leaf) is not FunctionType or leaf.__closure__ or code_key(leaf.__code__)!=leafspec['source']['code_key']:
        raise SourceMismatch('original temporal leaf callable differs from export')
    if _range(original) is not _RANGE:raise SourceMismatch('temporal original range binding differs')
    for source in (contract['source'],leafspec['source']):
        if sha256(Path(source['path']).read_bytes()).hexdigest()!=source['sha256']:
            raise SourceMismatch('original temporal source bytes differ from export')
    entry_state=_state(original);leaf_state=_state(leaf)
    shadow=FunctionType(original.__code__,original.__globals__,argdefs=original.__defaults__)
    shadow.__kwdefaults__=original.__kwdefaults__;signature=inspect.signature(shadow,follow_wrapped=False)
    stats=dict(native_calls=0,fallback_calls=0,reasons={})
    qualified=(type(selection) is dict and selection.get('decision')=='LOAD_MODEL_SELECTED_NATIVE'
        and selection.get('deployment_authorized') is True
        and selection.get('native_admission_sha256')==artifact.report['admission_sha256'])
    def fallback(reason,args,kwargs):
        stats['fallback_calls']+=1;stats['reasons'][reason]=stats['reasons'].get(reason,0)+1
        return original(*args,**kwargs)
    @wraps(original)
    def wrapped(*args,**kwargs):
        if not trial and not qualified:return fallback('no_validated_profitability_selection',args,kwargs)
        if not exclusive_inputs or threading.active_count()!=1 or sys.gettrace() is not None or sys.getprofile() is not None:
            return fallback('exclusive_synchronous_inputs_required',args,kwargs)
        monitoring=getattr(sys,'monitoring',None)
        if monitoring is not None and any(monitoring.get_tool(i) is not None for i in range(6)):
            return fallback('monitoring_observer_active',args,kwargs)
        if not _same_state(original,entry_state) or not _same_state(leaf,leaf_state) or _range(original) is not _RANGE or original.__globals__.get(leafspec['global_name']) is not leaf:
            return fallback('loop_leaf_defaults_or_range_changed',args,kwargs)
        try:bound=signature.bind(*args,**kwargs)
        except TypeError:return fallback('original_argument_binding',args,kwargs)
        bound.apply_defaults();count=bound.arguments[contract['trip_count_parameter']]
        if type(count) is not int or count!=contract['trip_count']:
            return fallback('outside_exact_count_specialization',args,kwargs)
        inputs={name:bound.arguments[outer] for name,outer in contract['parameter_map'].items()}
        try:result=artifact.run(inputs,count)
        except GuardFailure as error:return fallback(str(error),args,kwargs)
        stats['native_calls']+=1;return result
    wrapped.scar_stats=stats;wrapped.scar_artifact=artifact
    return wrapped


__all__=['NativeTemporalInvariant','GuardFailure','SourceMismatch','load','guarded_temporal_invariant']

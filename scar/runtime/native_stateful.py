"""Load the admitted stateful kernel without graphs, planners or compilation."""
from copy import deepcopy
import ctypes
from dataclasses import dataclass, field
from hashlib import sha256
import importlib
import math
from pathlib import Path
import platform
import struct
import sys
import threading

from .native_loop import (GuardFailure, _Array, _descriptor, _regular,
                          _file_hash_and_header, _cpu_available)
from .stateful_contract import inspect_stateful_bundle


class _Scalar(ctypes.Structure):
    _fields_=[('value',ctypes.c_double),('integer',ctypes.c_int64)]


class _Provider(ctypes.Structure):
    _fields_=[('loop',ctypes.c_void_p),('context',ctypes.c_void_p),('auxdata',ctypes.c_void_p)]


class _Info(ctypes.Structure):
    _fields_=_Provider._fields_+[('requires_pyapi',ctypes.c_ubyte),('no_fpe',ctypes.c_ubyte)]


_RUN=ctypes.CFUNCTYPE(ctypes.c_int32,ctypes.POINTER(_Array),ctypes.c_uint64,
    ctypes.POINTER(_Scalar),ctypes.c_uint64,ctypes.POINTER(_Provider),ctypes.c_uint64)


class _Table(ctypes.Structure):
    _fields_=[('abi_version',ctypes.c_uint32),('struct_size',ctypes.c_uint32),
        ('array_count',ctypes.c_uint64),('scalar_count',ctypes.c_uint64),
        ('provider_count',ctypes.c_uint64),('run',_RUN),('region_sha256',ctypes.c_char_p),
        ('plan_sha256',ctypes.c_char_p),('contract_sha256',ctypes.c_char_p)]


def _need(condition,message):
    if not condition: raise ValueError(message)


def _process(contract):
    cpu=contract['cpu']
    if (sys.platform!='linux' or platform.machine().lower() not in ('x86_64','amd64')
            or sys.byteorder!='little' or ctypes.sizeof(ctypes.c_void_p)!=8
            or ctypes.sizeof(_Array)!=64 or ctypes.sizeof(_Scalar)!=16
            or ctypes.sizeof(_Provider)!=24 or ctypes.sizeof(_Table)!=64):
        raise GuardFailure('stateful runtime requires its Linux x86_64 ABI')
    _cpu_available(cpu['ids'])


def _resolve_providers(contract):
    import numpy as np
    module=importlib.import_module('numpy._core._multiarray_umath')
    expected=contract['provider_runtime']; binary=Path(module.__file__).resolve()
    _need(np.__version__==expected['numpy_version']=='2.5.2'
          and int(module._get_ndarray_c_version())==expected['ndarray_c_abi'],
          'exact NumPy version/ndarray ABI differs')
    raw=binary.read_bytes()
    _need(dict(path=str(binary),sha256=sha256(raw).hexdigest(),bytes=len(raw))==expected['loaded_binary'],
          'loaded NumPy binary fingerprint differs')
    power=np.power; rows=[]; owners=[]
    for provider in contract['providers']:
        _need(type(power) is np.ufunc and power is module.power and power.__name__=='power',
              'original NumPy power ufunc required')
        dtypes,capsule=power._resolve_dtypes_and_context((np.dtype('float64'),)*3)
        _need(dtypes==(np.dtype('float64'),)*3,'NumPy power dtype resolution differs')
        power._get_strided_loop(capsule,fixed_strides=(8,0,8))
        getter=ctypes.pythonapi.PyCapsule_GetPointer
        getter.argtypes=[ctypes.py_object,ctypes.c_char_p]; getter.restype=ctypes.c_void_p
        address=getter(capsule,expected['capsule_abi'].encode())
        _need(address,'NumPy power capsule ABI differs')
        info=_Info.from_address(address)
        _need(info.loop and info.context and not info.requires_pyapi,'NumPy power loop requires unsupported Python API')
        rows.append(_Provider(info.loop,info.context,info.auxdata)); owners.append(capsule)
    return (_Provider*len(rows))(*rows),owners,power


@dataclass
class NativeStateful:
    directory: Path
    _contract: dict
    _report: dict
    _library: object
    _table: object
    _providers: object
    _owners: list
    _power: object
    _lock: object=field(default_factory=threading.Lock)

    @property
    def contract(self): return deepcopy(self._contract)

    @property
    def report(self): return deepcopy(self._report)

    def run(self,arrays,scalars):
        import numpy as np
        contract=self._contract
        if np.power is not self._power: raise GuardFailure('stateful original NumPy power provider changed')
        _cpu_available(contract['cpu']['ids'])
        array_specs=[row for row in contract['inputs'] if row['kind']=='array']
        scalar_specs=[row for row in contract['inputs'] if row['kind']=='scalar']
        if (type(arrays) is not dict or set(arrays)!={row['id'] for row in array_specs}
                or type(scalars) is not dict or set(scalars)!={row['id'] for row in scalar_specs}):
            raise GuardFailure('stateful input IDs differ')
        descriptors=[]; external=0
        for spec in array_specs:
            value=arrays[spec['id']]; descriptor=_descriptor(value,np)
            if (list(value.shape)!=spec['shape'] or value.nbytes!=spec['nbytes']
                    or not value.flags.c_contiguous or spec['written'] and not value.flags.writeable):
                raise GuardFailure('stateful shape/layout/writeability differs')
            for other in descriptors:
                if descriptor.lower_bound<other.upper_bound and other.lower_bound<descriptor.upper_bound:
                    raise GuardFailure('stateful external array storage aliases')
            descriptors.append(descriptor); external+=value.nbytes
        memory=contract['memory']
        if (external!=memory['external_array_bytes']
                or external+memory['workspace_payload_bytes']!=memory['total_region_array_payload_bytes']
                or memory['total_region_array_payload_bytes']>memory['memory_budget_bytes']
                or memory['total_region_array_payload_bytes']>sys.maxsize):
            raise GuardFailure('complete stateful region payload exceeds explicit budget')
        values=[]
        for spec in scalar_specs:
            value=scalars[spec['id']]
            if (type(value) not in (float,int) or type(value) is int and abs(value)>2**53
                    or not math.isfinite(value)):
                raise GuardFailure('finite exact scalar required')
            values.append(_Scalar(float(value),int(type(value) is int)))
        if not self._lock.acquire(blocking=False): raise GuardFailure('stateful provider capsule already in use')
        try:
            status=self._table.contents.run((_Array*len(descriptors))(*descriptors),len(descriptors),
                (_Scalar*len(values))(*values),len(values),self._providers,len(self._owners))
        finally: self._lock.release()
        if status:
            reason={1:'arguments/alias',2:'shape/layout',3:'floating-point/nonfinite',4:'CPU affinity',
                    5:'allocation',6:'provider',7:'payload budget',8:'scalar exactness'}.get(status,str(status))
            raise GuardFailure('stateful precommit guard: '+reason)
        return None


def load(bundle,*,expected_admission_sha256):
    metadata=inspect_stateful_bundle(bundle,expected_admission_sha256=expected_admission_sha256)
    contract=metadata['contract']; native=contract['native']; directory=Path(bundle).resolve()
    core={'runtime.json','admission.json',native['library']}; members={p.name for p in directory.iterdir()}
    _need(core<=members<=core|{'export-report.json'},'stateful bundle has missing core files or unknown members')
    path=directory/native['library']; digest,header=_file_hash_and_header(path)
    _need(digest==native['library_sha256'],'stateful library byte SHA256 differs')
    _process(contract)
    _need(len(header)>=64 and header[:6]==b'\x7fELF\x02\x01' and struct.unpack_from('<H',header,16)[0]==3
          and struct.unpack_from('<H',header,18)[0]==62,'stateful library requires ELF64 little-endian x86_64')
    providers,owners,power=_resolve_providers(contract)
    library=ctypes.CDLL(str(path)); getter=library.scar_get_stateful_table_v1
    getter.argtypes=[]; getter.restype=ctypes.POINTER(_Table); table=getter()
    _need(table and table.contents.abi_version==1 and table.contents.struct_size==ctypes.sizeof(_Table)
          and table.contents.run,'stateful native function table ABI differs')
    expected=(sum(row['kind']=='array' for row in contract['inputs']),sum(row['kind']=='scalar' for row in contract['inputs']),len(contract['providers']))
    _need((table.contents.array_count,table.contents.scalar_count,table.contents.provider_count)==expected,'stateful native table counts differ')
    for key in ('region_sha256','plan_sha256','contract_sha256'):
        _need(getattr(table.contents,key)==native['ids'][key].encode(),'stateful embedded native identity differs: '+key)
    memory=contract['memory']
    report=dict(schema='scar.loaded-stateful-runtime',schema_version=1,**native['ids'],profile=contract['profile'],
        provider_runtime=contract['provider_runtime'],cpu_ids=contract['cpu']['ids'],cpu_threads=contract['cpu']['threads'],
        **memory,admission_sha256=expected_admission_sha256,library_sha256=digest,
        source_guard_scope='INSTALL_AND_FIRST_RECEIVER_BINDING_SOURCE_SHA_ONCE_THEN_LIVE_CODE_DEFAULT_CALLEE_PROVIDER_AND_INPUT_GUARDS',
        planning_replayed_at_runtime=False,native_recompiled=False,original_kernel_bytes_unchanged=True,
        fresh_process_cost_qualified=False)
    return NativeStateful(directory,contract,report,library,table,providers,owners,power)


__all__=['NativeStateful','GuardFailure','load']

"""Producer-pinned, unmeasured composition of the original stateful ABI1.

The optional base has no measured selection and is usable only through this
explicit trial profile. Its preflight, provider ownership/lock and per-step
transaction remain the existing NativeStateful implementation.
"""
from copy import deepcopy
import ctypes
from dataclasses import dataclass,replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from . import native_stateful as step_runtime
from .native_loop import _Array,_small_raw,_json,_file_hash_and_header,GuardFailure
from .stateful_contract import validate_contract as validate_step
from .loop_contract import _source,_digest

SCHEMA='scar.stateful-loop-runtime-contract'
ADMISSION='scar.stateful-loop-runtime-admission'
PROFILE='UNMEASURED_TRIAL_STATEFUL_REPEAT_ABI1'
FAMILY='ORDERED_TRANSACTIONAL_STATEFUL_REPEAT_ABI1_V1'
WORKSPACE_PROFILE='UNMEASURED_TRIAL_STATEFUL_WORKSPACE_REPEAT_ABI1'
WORKSPACE_FAMILY='ORDERED_TRANSACTIONAL_STATEFUL_WORKSPACE_REPEAT_ABI1_V1'
GUARD_REUSE_PROFILE='UNMEASURED_TRIAL_STATEFUL_WORKSPACE_FINITE_REUSE_ABI1'
GUARD_REUSE_FAMILY='ORDERED_TRANSACTIONAL_STATEFUL_WORKSPACE_FINITE_REUSE_ABI1_V1'
_WORKSPACE_PROFILES=(WORKSPACE_PROFILE,GUARD_REUSE_PROFILE)
TRUST='EXPLICIT_EXPECTED_ADMISSION_SHA256_FROM_ORIGINAL_SOURCE_LOOP_EXPORTER_NOT_SIGNATURES'


def _need(condition,message):
    if not condition:raise ValueError(message)


def validate_contract(value):
    _need(type(value)is dict and(value.get('schema'),value.get('schema_version'))==(SCHEMA,1)
        and (value['profile'],value['execution_family'])in(
            (PROFILE,FAMILY),(WORKSPACE_PROFILE,WORKSPACE_FAMILY),
            (GUARD_REUSE_PROFILE,GUARD_REUSE_FAMILY))
        and value['cost_status']=='UNKNOWN'and value['deployment_authorized']is False
        and value['trial_only']is True,'explicit unmeasured stateful loop contract required')
    _source(value['source']);_source(value['step_source'])
    step=validate_step(value['step_contract'],require_selection=False)
    _need(step['selection']is None and step['source']==value['step_source'],
        'trial step must retain original source without a fabricated measured selection')
    parameters=value['parameters'];mapping=value['parameter_map'];count=value['trip_count']
    _need(type(parameters)is list and parameters and len(set(parameters))==len(parameters)
        and all(type(n)is str and n.isidentifier()for n in parameters)
        and value['trip_count_parameter']in parameters
        and type(count)is int and 0<=count<2**64
        and type(mapping)is dict and set(mapping)==set(step['parameters'])
        and all(n in parameters for n in mapping.values())
        and type(value['step_global_name'])is str and value['step_global_name'].isidentifier(),
        'original fixed caller/step/count parameter interface required')
    call=value['call_arguments']
    _need(type(call)is dict and set(call)=={'positional_parameters','keyword_parameters'}
        and type(call['positional_parameters'])is list and type(call['keyword_parameters'])is list
        and all(n in parameters for n in call['positional_parameters'])
        and all(type(pair)is list and len(pair)==2 and type(pair[0])is str
            and pair[0].isidentifier()and pair[1]in parameters for pair in call['keyword_parameters'])
        and len({pair[0]for pair in call['keyword_parameters']})==len(call['keyword_parameters']),
        'original positional/keyword CALL syntax required')
    native=value['native']
    workspace=value['profile']in _WORKSPACE_PROFILES
    if workspace:
        scope=value.get('workspace')
        _need(type(scope)is dict and set(scope)=={'plan_sha256','execution_plan_sha256','slot_capacities_bytes'}
            and _digest(scope['plan_sha256'])and _digest(scope['execution_plan_sha256'])
            and type(scope['slot_capacities_bytes'])is list and len(scope['slot_capacities_bytes'])<=4096
            and all(type(cap)is int and cap>=0 and cap%8==0 for cap in scope['slot_capacities_bytes'])
            and sum(scope['slot_capacities_bytes'])==step['memory']['workspace_payload_bytes']
            and value.get('workspace_allocations_hoisted')is True
            and value.get('team_setup_hoisted')is False
            and value.get('workspace_or_team_setup_hoisted')is True,
            'workspace profile requires exact plan/capacities and allocation-only hoisting')
    else:
        _need('workspace'not in value and value.get('workspace_or_team_setup_hoisted')is False
            and value.get('workspace_allocations_hoisted',False)is False
            and value.get('team_setup_hoisted',False)is False,
            'original repeat profile cannot carry workspace or hoisted setup')
    if value['profile']==GUARD_REUSE_PROFILE:
        reuse=value.get('guard_reuse')
        arrays=sum(row['kind']=='array'for row in step['inputs'])
        _need(type(reuse)is dict and set(reuse)=={'property_plan_sha256',
            'parent_execution_plan_sha256','input_array_indices','policy'}
            and _digest(reuse['property_plan_sha256'])and _digest(reuse['parent_execution_plan_sha256'])
            and arrays>0 and type(reuse['input_array_indices'])is list
            and all(type(i)is int for i in reuse['input_array_indices'])
            and reuse['input_array_indices']==list(range(arrays)) and count>1 and not step['providers']
            and reuse['policy']=='FIRST_INPUT_SCAN_THEN_SUCCESS_FINITE_PROPERTY_INDUCTION'
            and value.get('input_finite_scans_reused')is True,
            'finite-reuse profile requires the exact property plan and guarded input set')
        execution=dict(schema='scar.workspace-guard-reuse-execution-plan',schema_version=1,
            parent_execution_plan_sha256=reuse['parent_execution_plan_sha256'],
            property_plan_sha256=reuse['property_plan_sha256'])
        _need(sha256(_json_bytes(execution)).hexdigest()==value['workspace']['execution_plan_sha256'],
            'finite property and parent controller must identify the new execution plan')
    else:
        _need('guard_reuse'not in value and value.get('input_finite_scans_reused',False)is False,
            'original profiles cannot silently reuse input finite guards')
    _need(native['stateful_abi_version']==native['repeat_abi_version']==1
        and native['symbol']==('scar_workspace_repeat_stateful_v1'if workspace else'scar_repeat_stateful_v1'),
        'profile-matched original stateful/repeat ABI1 required')
    for row in(native['base'],native['repeat']):
        _need(type(row)is dict and Path(row['library']).name==row['library']
            and row['library']not in('runtime.json','admission.json','export-report.json')
            and _digest(row['sha256'])and type(row['bytes'])is int and row['bytes']>=64,
            'local pinned original ELF required')
    _need(native['base']['library']==step['native']['library']
        and native['base']['sha256']==step['native']['library_sha256']
        and native['base_ids']==step['native']['ids'], 'repeat/base stateful identities differ')
    _need(value['commit_policy']=='EACH_SUCCESSFUL_ORIGINAL_ABI_CALL_COMMITS_BEFORE_NEXT_STEP'
        and value['failure_policy']=='RESUME_ONLY_UNCOMMITTED_ORIGINAL_SUFFIX_WITH_DYNAMIC_CALLEE_LOOKUP'
        and type(value['conditions'])is list and value['conditions']
        and value['source_loop_replacement_authority']=='ORIGINAL_F4_LOOP_AND_STEP_REPLAY_AT_EXPORT',
        'per-step transaction/progress/source contract required')
    return value


def implementation_sha256():
    root=Path(__file__).resolve().parents[1]
    names=('runtime/native_stateful_repeat.py','runtime/stateful_loop_guard.py','runtime/native_stateful.py',
        'runtime/stateful_guard.py','runtime/stateful_contract.py','runtime/native_loop.py',
        'runtime/loop_contract.py','trace/code_snapshot.py')
    rows=[(name,sha256((root/name).read_bytes()).hexdigest())for name in names]
    return sha256(_json_bytes(rows)).hexdigest()


def _json_bytes(value):
    import json
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _base(directory,contract):
    """Light trial factory; reuse the original runtime and native table checks."""
    step_runtime._process(contract)
    providers,owners,power=step_runtime._resolve_providers(contract)
    library=ctypes.CDLL(str(directory/contract['native']['library']))
    getter=library.scar_get_stateful_table_v1;getter.argtypes=[]
    getter.restype=ctypes.POINTER(step_runtime._Table);table=getter()
    _need(table and table.contents.abi_version==1 and table.contents.struct_size==ctypes.sizeof(step_runtime._Table)
        and table.contents.run,'original step table ABI differs')
    _need((table.contents.array_count,table.contents.scalar_count,table.contents.provider_count)==(
        sum(row['kind']=='array'for row in contract['inputs']),sum(row['kind']=='scalar'for row in contract['inputs']),
        len(contract['providers'])),'original step table input counts differ')
    for name in('region_sha256','plan_sha256','contract_sha256'):
        _need(getattr(table.contents,name)==contract['native']['ids'][name].encode(),
            'original step embedded identity differs: '+name)
    report=dict(**contract['native']['ids'],trial_only=True,cost_status='UNKNOWN',deployment_authorized=False)
    return step_runtime.NativeStateful(directory,contract,report,library,table,providers,owners,power)


class RepeatGuardFailure(GuardFailure):
    def __init__(self,requested,completed,status,reason):
        super().__init__(reason)
        self.requested=requested;self.completed=completed;self.status=status


@dataclass
class NativeStatefulRepeat:
    directory:Path
    base:object
    _contract:dict
    _report:dict
    _library:object
    _repeat:object

    @property
    def loop_contract(self):return deepcopy(self._contract)
    @property
    def report(self):return deepcopy(self._report)

    def run(self,arrays,scalars,repetitions):
        if type(repetitions)is not int or not 0<=repetitions<2**64:
            raise ValueError('exact nonnegative uint64 repetition count required')
        if repetitions==0:
            result=dict(requested=0,completed=0,native_calls=0,status=0)
            self._report['last_execution']=result;return dict(result)
        if self._contract['profile']in _WORKSPACE_PROFILES and repetitions!=self._contract['trip_count']:
            self._report['last_execution']=dict(requested=repetitions,completed=0,native_calls=0,status=None)
            raise RepeatGuardFailure(repetitions,0,None,'count differs from exact workspace plan')
        progress=ctypes.c_uint64();status=None
        def invoke(*args):
            nonlocal status
            status=self._repeat(self.base._table,*args,repetitions,ctypes.byref(progress))
            return status
        facade=replace(self.base,_table=SimpleNamespace(contents=SimpleNamespace(run=invoke)))
        try:facade.run(arrays,scalars)
        except GuardFailure as error:
            self._report['last_execution']=dict(requested=repetitions,completed=progress.value,
                native_calls=progress.value,status=status)
            raise RepeatGuardFailure(repetitions,progress.value,status,str(error))from error
        _need(progress.value==repetitions,'repeat returned success without complete committed progress')
        result=dict(requested=repetitions,completed=progress.value,native_calls=progress.value,status=0)
        self._report['last_execution']=result;return dict(result)


def _admit(bundle,expected_admission_sha256):
    _need(_digest(expected_admission_sha256),'explicit original loop-export admission pin required')
    directory=Path(bundle).resolve();raw=_small_raw(directory/'admission.json');admission=_json(raw)
    _need(sha256(raw).hexdigest()==expected_admission_sha256
        and(admission.get('schema'),admission.get('schema_version'))==(ADMISSION,1)
        and admission['trust_policy']==TRUST and admission['profile']in(PROFILE,*_WORKSPACE_PROFILES)
        and admission['runtime_implementation_sha256']==implementation_sha256(),
        'original unmeasured loop producer/runtime admission differs')
    raw=_small_raw(directory/'runtime.json')
    _need(sha256(raw).hexdigest()==admission['runtime_sha256'],'loop runtime contract bytes differ')
    contract=validate_contract(_json(raw));native=contract['native']
    _need(native==admission['native']and admission['profile']==contract['profile'],
        'loop native bytes/ABI/profile differ from admission')
    members={'runtime.json','admission.json',native['base']['library'],native['repeat']['library']}
    _need(members<={p.name for p in directory.iterdir()}<=members|{'export-report.json'},
        'unexpected stateful loop compact bundle members')
    import struct
    for row in(native['base'],native['repeat']):
        path=directory/row['library'];digest,header=_file_hash_and_header(path)
        _need(digest==row['sha256']and path.stat().st_size==row['bytes']and len(header)>=64
            and header[:6]==b'\x7fELF\x02\x01'and struct.unpack_from('<H',header,16)[0]==3
            and struct.unpack_from('<H',header,18)[0]==62,'original repeat/base ELF bytes or ABI differ')
    return directory,contract,admission


def inspect_bundle(bundle,*,expected_admission_sha256):
    """Admit compact metadata/ELF bytes without CDLL or provider construction."""
    directory,contract,admission=_admit(bundle,expected_admission_sha256)
    return dict(directory=str(directory),contract=deepcopy(contract),admission=deepcopy(admission),
        admission_sha256=expected_admission_sha256,metadata_only=True,
        library_bytes_checked=True,library_loaded=False)


def load(bundle,*,expected_admission_sha256):
    directory,contract,admission=_admit(bundle,expected_admission_sha256)
    native=contract['native']
    base=_base(directory,contract['step_contract'])
    library=ctypes.CDLL(str(directory/native['repeat']['library']))
    if contract['profile']in _WORKSPACE_PROFILES:
        getter=library.scar_workspace_plan_sha256_v1;getter.argtypes=[];getter.restype=ctypes.c_char_p
        _need(getter()==contract['workspace']['execution_plan_sha256'].encode(),
            'workspace embedded execution plan differs from compact contract')
    run=getattr(library,native['symbol'])
    run.argtypes=[ctypes.POINTER(step_runtime._Table),ctypes.POINTER(_Array),ctypes.c_uint64,
        ctypes.POINTER(step_runtime._Scalar),ctypes.c_uint64,ctypes.POINTER(step_runtime._Provider),ctypes.c_uint64,
        ctypes.c_uint64,ctypes.POINTER(ctypes.c_uint64)]
    run.restype=ctypes.c_int32
    return NativeStatefulRepeat(directory,base,contract,dict(admission_sha256=expected_admission_sha256,
        cost_status='UNKNOWN',trial_only=True,deployment_authorized=False,last_execution=None),library,run)


__all__=['load','inspect_bundle','validate_contract','implementation_sha256','NativeStatefulRepeat','RepeatGuardFailure',
    'SCHEMA','ADMISSION','PROFILE','FAMILY','WORKSPACE_PROFILE','WORKSPACE_FAMILY','TRUST']

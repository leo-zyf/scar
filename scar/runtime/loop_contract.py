"""Small, versioned deployment contracts, independent of graph and compiler IR."""
from hashlib import sha256
import json
from pathlib import Path
import re

SCHEMA = 'scar.loop-runtime-contract'
ADMISSION_SCHEMA = 'scar.loop-runtime-admission'
TRUST_POLICY = 'EXPLICIT_EXPECTED_ADMISSION_SHA256_FROM_BUILD_PRODUCER_NOT_A_SIGNATURE'
ASSUMPTIONS = frozenset(('SYNCHRONOUS_CPU_EXECUTION', 'NO_FOREIGN_CONCURRENT_INPUT_WRITERS',
                         'FP_STATUS_FLAGS_NOT_USED_AS_PROGRAM_OUTPUT'))
IDENTITIES = frozenset(('build_id','loop_sha256','strategy_sha256','execution_plan_sha256','contract_sha256'))


def canonical_sha256(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _need(condition, message):
    if not condition: raise ValueError(message)


def _keys(value, expected, name):
    _need(type(value) is dict and set(value)==set(expected), 'invalid '+name+' fields')


def _digest(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def _integer(value, lower=0, upper=(1<<63)-1):
    return type(value) is int and lower<=value<=upper


def _source(value):
    _keys(value,('path','sha256','code_key','qualname'),'source')
    _need(type(value['path']) is str and Path(value['path']).is_absolute()
          and _digest(value['sha256']) and type(value['code_key']) is str
          and re.fullmatch('pycode-v2:[0-9a-f]{64}',value['code_key']) is not None
          and type(value['qualname']) is str and value['qualname'], 'invalid source/code identity')


def _identities(value):
    _keys(value,IDENTITIES,'native identities')
    _need(all(_digest(value[k]) for k in IDENTITIES),'invalid native identity hashes')


def validate_contract(value):
    _keys(value,('schema','schema_version','profile','native','source','parameters','state_parameter',
        'trip_count_parameter','callee_guards','array','trip_count','memory_budget_bytes','cpu',
        'external_assumptions','floating_environment_policy','selection'),'runtime contract')
    _need(value['schema']==SCHEMA and type(value['schema_version']) is int
          and value['schema_version']==1 and value['profile']=='cpu-f64-rank2-loop-v1',
          'unsupported runtime contract version/profile')
    native=value['native']
    _keys(native,('library','library_sha256','loop_abi_version','ids','contract'),'native')
    name=native['library']
    _need(type(name) is str and name not in ('','.','..') and Path(name).name==name
          and '/' not in name and '\\' not in name and _digest(native['library_sha256'])
          and type(native['loop_abi_version']) is int and native['loop_abi_version']==1,
          'invalid native library reference or ABI')
    _identities(native['ids']);_source(value['source'])
    parameters=value['parameters']
    _need(type(parameters) is list and len(parameters)==2 and all(type(p) is str and p.isidentifier() for p in parameters)
          and len(set(parameters))==2 and {value['state_parameter'],value['trip_count_parameter']}==set(parameters),
          'runtime boundary requires one array and one exact integer parameter')
    _need(_integer(value['trip_count'],1) and _integer(value['memory_budget_bytes']),
          'positive exact loop count and explicit memory budget required')
    array=value['array']
    _keys(array,('shape','dtype','device','layout','alignment_bytes','final_bytes'),'array')
    shape=array['shape']
    _need(type(shape) is list and len(shape)==2 and all(_integer(n) for n in shape)
          and array['dtype']=='float64' and array['device']=='cpu' and array['layout']=='C'
          and type(array['alignment_bytes']) is int and array['alignment_bytes']==8,
          'exact native float64 rank-two C array required')
    size=shape[0]*shape[1]*8
    _need(_integer(array['final_bytes']) and size==array['final_bytes']<=value['memory_budget_bytes'],
          'runtime final payload or budget differs')
    cpu=value['cpu']
    _keys(cpu,('ids','threads','architecture','os','pointer_bits','byteorder','required_isa'),'CPU')
    _need(type(cpu['ids']) is list and cpu['ids'] and all(_integer(n,0,(1<<20)-1) for n in cpu['ids'])
          and len(set(cpu['ids']))==len(cpu['ids']) and _integer(cpu['threads'],1)
          and cpu['threads']==len(cpu['ids']) and cpu['architecture']=='x86_64' and cpu['os']=='linux'
          and type(cpu['pointer_bits']) is int and cpu['pointer_bits']==64 and cpu['byteorder']=='little'
          and cpu['required_isa'] in ('SSE2_X86_64','BASELINE_X86_64'),'unsupported CPU/process contract')
    guards=value['callee_guards']
    _need(type(guards) is list and 1<=len(guards)<=512,'bounded nested callable guards required')
    paths=set()
    for guard in guards:
        _keys(guard,('environment_path','source'),'callee guard');_source(guard['source'])
        path=guard['environment_path']
        _need(type(path) is list and 1<=len(path)<=2 and all(type(p) is str and p.isidentifier() for p in path)
              and tuple(path) not in paths,'unique function-globals paths required')
        paths.add(tuple(path))
    roots={p for p in paths if len(p)==1}
    _need(len(roots)==1 and all(len(p)==1 or p[:1] in roots for p in paths),
          'leaf guards must refer to the original body globals')
    assumptions=value['external_assumptions']
    _need(type(assumptions) is list and all(type(a) is str for a in assumptions)
          and len(assumptions)==len(ASSUMPTIONS) and set(assumptions)==ASSUMPTIONS,
          'complete external assumption contract required')
    _need(value['floating_environment_policy']=='PRESERVE_CALLER_AND_WORKER_ENVIRONMENT_NOT_NUMPY_STATUS_FLAG_EQUIVALENCE',
          'unsupported floating environment policy')
    contract=native['contract']
    version=contract.get('schema_version') if type(contract) is dict else None
    fields=('schema','schema_version','loop_sha256','strategy_sha256','execution_plan_sha256','trip_count',
            'state_shape','output_shape','final_bytes','memory_budget_bytes')
    _need(type(version) is int and version in (1,2),'unsupported native contract')
    _keys(contract,(*fields,*(['layout_sha256'] if version==2 else [])),'native contract')
    _need(contract['schema']=='scar.native-loop-contract' and all(contract[k]==native['ids'][k]
          for k in ('loop_sha256','strategy_sha256','execution_plan_sha256'))
          and contract['trip_count']==value['trip_count'] and contract['state_shape']==shape
          and contract['output_shape']==shape and contract['final_bytes']==size
          and contract['memory_budget_bytes']==value['memory_budget_bytes']
          and canonical_sha256(contract)==native['ids']['contract_sha256'],
          'runtime fields differ from the native embedded contract')
    if version==2:_need(_digest(contract['layout_sha256']) and cpu['required_isa']=='SSE2_X86_64',
                        'layout contract needs its hash and SSE2 implementation')
    else:_need(cpu['required_isa']=='BASELINE_X86_64','scalar contract cannot assert SIMD layout')
    selection=value['selection']
    _need(type(selection) is dict and selection.get('decision')=='TRANSFORM_UNDER_CONTRACT'
          and selection.get('selected') is True and type(selection.get('artifact')) is dict
          and all(selection['artifact'].get(k)==native['ids'][k]
                  for k in ('build_id','loop_sha256','strategy_sha256'))
          and selection.get('external_assumptions')==assumptions,
          'measured selection must match the exact artifact and full assumption ledger')
    return value


def validate_admission(value):
    _keys(value,('schema','schema_version','runtime_contract_sha256','library_sha256','native_ids',
                 'producer','evidence','trust_policy'),'admission')
    _need(value['schema']==ADMISSION_SCHEMA and type(value['schema_version']) is int
          and value['schema_version']==1 and value['trust_policy']==TRUST_POLICY,
          'unsupported admission/trust contract')
    _need(_digest(value['runtime_contract_sha256']) and _digest(value['library_sha256']),
          'admission must bind runtime bytes and library bytes')
    _identities(value['native_ids'])
    producer=value['producer'];_keys(producer,('name','version','exporter_sha256'),'producer')
    _need(producer['name']=='SCAR' and type(producer['version']) is str and producer['version']
          and _digest(producer['exporter_sha256']),'explicit build producer required')
    evidence=value['evidence']
    _need(type(evidence) is list and 1<=len(evidence)<=4096,'bounded build evidence receipts required')
    for row in evidence:
        _keys(row,('role','path','sha256','bytes'),'evidence')
        _need(type(row['role']) is str and row['role'] and type(row['path']) is str
              and Path(row['path']).is_absolute() and _digest(row['sha256']) and _integer(row['bytes']),
              'invalid build evidence receipt')
    return value


__all__=['SCHEMA','ADMISSION_SCHEMA','TRUST_POLICY','ASSUMPTIONS','IDENTITIES',
         'canonical_sha256','validate_contract','validate_admission']

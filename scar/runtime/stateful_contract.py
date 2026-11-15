"""Small stateful deployment contracts; no compiler, graph or planner imports."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import re

from .loop_contract import canonical_sha256, _keys, _need, _source, _digest, _integer

SCHEMA = 'scar.stateful-runtime-contract'
ADMISSION_SCHEMA = 'scar.stateful-runtime-admission'
TRUST_POLICY = 'EXPLICIT_EXPECTED_ADMISSION_SHA256_FROM_BUILD_PRODUCER_NOT_A_SIGNATURE'
IDENTITIES = frozenset(('build_id','region_sha256','plan_sha256','contract_sha256'))
ASSUMPTIONS = frozenset(('SYNCHRONOUS_CPU_EXECUTION', 'NO_FOREIGN_CONCURRENT_STATE_READERS_OR_WRITERS',
                         'FP_STATUS_FLAGS_NOT_USED_AS_PROGRAM_OUTPUT'))
MEMORY_SCOPE = 'ALL_REGION_ARRAY_PAYLOAD_EXTERNAL_STATE_PLUS_FIXED_WORKSPACE'
FP_POLICY = 'PRESERVE_CALLER_AND_WORKER_ENVIRONMENT_NOT_NUMPY_STATUS_FLAG_EQUIVALENCE'


def _ids(value):
    _keys(value, IDENTITIES, 'stateful identities')
    _need(all(_digest(v) for v in value.values()), 'invalid stateful native identity')


def validate_contract(value, *, require_selection=True):
    _keys(value, ('schema','schema_version','profile','native','source','parameters','inputs',
        'callee_guards','providers','cpu','memory','provider_runtime','guards','external_assumptions',
        'floating_environment_policy','selection'), 'stateful runtime contract')
    _need(value['schema']==SCHEMA and type(value['schema_version']) is int and value['schema_version']==1
          and value['profile']=='cpu-f64-rank2-stateful-v1', 'unsupported stateful runtime profile')
    native=value['native']; _keys(native, ('library','library_sha256','stateful_abi_version','ids','contract'), 'native')
    name=native['library']
    _need(type(name) is str and name not in ('','.','..') and Path(name).name==name and '/' not in name
          and '\\' not in name and _digest(native['library_sha256'])
          and type(native['stateful_abi_version']) is int and native['stateful_abi_version']==1,
          'invalid stateful native library/ABI')
    _ids(native['ids']); _source(value['source'])
    parameters=value['parameters']
    _need(type(parameters) is list and parameters and len(set(parameters))==len(parameters)
          and all(type(p) is str and p.isidentifier() for p in parameters), 'ordered original parameters required')
    inputs=value['inputs']; _need(type(inputs) is list and 0<len(inputs)<=4096, 'bounded state inputs required')
    ids=set(); paths=set(); external=0; array_shapes={}
    for row in inputs:
        _keys(row, ('id','path','kind','shape','nbytes','written'), 'state input')
        path=row['path']
        _need(type(row['id']) is str and row['id'] and row['id'] not in ids and type(path) is list and path
              and path[0] in parameters and all(type(p) is str and p.isidentifier() for p in path)
              and tuple(path) not in paths and type(row['written']) is bool, 'unique guarded input paths required')
        ids.add(row['id']); paths.add(tuple(path))
        if row['kind']=='array':
            shape=row['shape']; _need(type(shape) is list and len(shape)==2 and all(_integer(n) for n in shape), 'rank-two input shape required')
            size=shape[0]*shape[1]*8
            _need(_integer(row['nbytes']) and row['nbytes']==size, 'state input payload differs')
            external+=size; array_shapes[row['id']]=shape
        else:
            _need(row['kind']=='scalar' and row['shape'] is None and row['nbytes'] is None and not row['written'], 'finite scalar input required')
    _need(any(row['written'] for row in inputs), 'at least one original state write required')
    guards=value['callee_guards']; _need(type(guards) is list and len(guards)<=4096, 'bounded callable guards required')
    seen=set()
    for guard in guards:
        _keys(guard, ('path','kind','source'), 'stateful callee'); _source(guard['source'])
        path=guard['path']
        _need(guard['kind']=='STATICMETHOD' and type(path) is list and path and all(type(p) is str and p.isidentifier() for p in path)
              and tuple(path) not in seen, 'unique ordinary staticmethod path required'); seen.add(tuple(path))
    providers=value['providers']; _need(type(providers) is list and len(providers)<=256, 'bounded provider guards required')
    seen=set()
    for row in providers:
        _keys(row, ('id','module','symbol','lookup_path','semantics'), 'provider')
        _need(type(row['id']) is str and row['id'] not in seen and row['module']=='numpy' and row['symbol']=='power'
              and row['semantics']=='NUMPY_UFUNC_POWER_FLOAT64_EXACT_PYTHON_INT_TWO_NOT_MULTIPLICATION_REWRITE'
              and type(row['lookup_path']) is list and row['lookup_path']
              and all(type(p) is str and p.isidentifier() for p in row['lookup_path']), 'original NumPy power provider required')
        seen.add(row['id'])
    runtime=value['provider_runtime']
    _keys(runtime, ('API_stability','capsule_abi','dtype_signature','execution','fixed_strides','loaded_binary',
                    'ndarray_c_abi','numpy_version','requires_pyapi'), 'provider runtime')
    binary=runtime['loaded_binary']; _keys(binary, ('path','sha256','bytes'), 'NumPy binary')
    _need(runtime['numpy_version']=='2.5.2' and runtime['capsule_abi']=='numpy_1.24_ufunc_call_info'
          and runtime['ndarray_c_abi']==33554432 and runtime['dtype_signature']==['float64']*3
          and runtime['fixed_strides']==[8,0,8] and runtime['requires_pyapi'] is False
          and runtime['API_stability']=='EXPERIMENTAL_EXACT_NUMPY_VERSION_AND_BINARY_PIN'
          and runtime['execution']=='ORIGINAL_NUMPY_UFUNC_STRIDED_LOOP_SERIAL_PROVIDER_STAGE'
          and type(binary['path']) is str and Path(binary['path']).is_absolute() and _digest(binary['sha256'])
          and _integer(binary['bytes'],1), 'exact supported NumPy native provider contract required')
    memory=value['memory']; _keys(memory, ('external_array_bytes','workspace_payload_bytes',
        'total_region_array_payload_bytes','memory_budget_bytes','scope','policy'), 'region memory')
    _need(all(_integer(memory[k]) for k in ('external_array_bytes','workspace_payload_bytes','total_region_array_payload_bytes','memory_budget_bytes'))
          and memory['external_array_bytes']==external and memory['total_region_array_payload_bytes']==external+memory['workspace_payload_bytes']
          and memory['total_region_array_payload_bytes']<=memory['memory_budget_bytes']
          and memory['scope']==MEMORY_SCOPE and memory['policy']=='UNBOUND', 'complete region payload budget required')
    cpu=value['cpu']; _keys(cpu, ('ids','threads','architecture','os','pointer_bits','byteorder','required_isa'), 'CPU')
    _need(type(cpu['ids']) is list and cpu['ids'] and all(_integer(n,0,(1<<20)-1) for n in cpu['ids'])
          and len(set(cpu['ids']))==len(cpu['ids']) and _integer(cpu['threads'],1) and cpu['threads']==len(cpu['ids'])
          and cpu['architecture']=='x86_64' and cpu['os']=='linux' and cpu['pointer_bits']==64
          and cpu['byteorder']=='little' and cpu['required_isa']=='BASELINE_X86_64', 'unsupported CPU/process profile')
    assumptions=value['external_assumptions']
    _need(type(assumptions) is list and len(assumptions)==len(ASSUMPTIONS) and set(assumptions)==ASSUMPTIONS
          and value['floating_environment_policy']==FP_POLICY, 'complete external state/fenv assumptions required')
    _need(type(value['guards']) is list and value['guards'] and all(type(g) is str for g in value['guards']), 'guard ledger required')
    core=native['contract']; _keys(core, ('region_sha256','plan_sha256','provider_runtime'), 'embedded native contract')
    _need(core['region_sha256']==native['ids']['region_sha256'] and core['plan_sha256']==native['ids']['plan_sha256']
          and core['provider_runtime']==runtime and canonical_sha256(core)==native['ids']['contract_sha256'], 'native contract identity differs')
    selection=value['selection']
    if require_selection:
        _need(type(selection) is dict and selection.get('decision')=='TRANSFORM_UNDER_CONTRACT' and selection.get('selected') is True
              and type(selection.get('artifact')) is dict and all(selection['artifact'].get(k)==native['ids'][k] for k in ('build_id','plan_sha256'))
              and selection.get('input_shapes')==array_shapes and selection.get('external_assumptions')==assumptions,
              'measured selection must bind this exact stateful artifact, shapes and assumptions')
    else: _need(selection is None or type(selection) is dict, 'invalid optional selection')
    return value


def validate_admission(value):
    _keys(value, ('schema','schema_version','runtime_contract_sha256','library_sha256','native_ids','producer','evidence','trust_policy'), 'stateful admission')
    _need(value['schema']==ADMISSION_SCHEMA and type(value['schema_version']) is int and value['schema_version']==1 and value['trust_policy']==TRUST_POLICY
          and _digest(value['runtime_contract_sha256']) and _digest(value['library_sha256']), 'invalid stateful admission')
    _ids(value['native_ids']); producer=value['producer']; _keys(producer, ('name','version','exporter_sha256'), 'producer')
    _need(producer['name']=='SCAR' and type(producer['version']) is str and producer['version'] and _digest(producer['exporter_sha256']), 'build producer required')
    _need(type(value['evidence']) is list and 0<len(value['evidence'])<=4096, 'bounded original evidence required')
    for row in value['evidence']:
        _keys(row, ('role','path','sha256','bytes'), 'evidence')
        _need(type(row['role']) is str and row['role'] and type(row['path']) is str and Path(row['path']).is_absolute()
              and _digest(row['sha256']) and _integer(row['bytes']), 'invalid evidence receipt')
    return value


def inspect_stateful_bundle(bundle, *, expected_admission_sha256):
    from .native_loop import _small_raw, _json
    _need(_digest(expected_admission_sha256), 'explicit expected admission SHA256 required')
    directory=Path(bundle).resolve(); raw=_small_raw(directory/'admission.json')
    _need(sha256(raw).hexdigest()==expected_admission_sha256, 'admission SHA256 differs from explicit trust pin')
    admission=validate_admission(_json(raw)); raw=_small_raw(directory/'runtime.json')
    _need(sha256(raw).hexdigest()==admission['runtime_contract_sha256'], 'runtime contract byte SHA256 differs')
    contract=validate_contract(_json(raw))
    _need(contract['native']['ids']==admission['native_ids'] and contract['native']['library_sha256']==admission['library_sha256'], 'runtime/native admission identities differ')
    return deepcopy(dict(contract=contract,admission=admission,admission_sha256=expected_admission_sha256,
        metadata_only=True,library_loaded=False,library_bytes_checked=False,execution_contract_must_be_revalidated=True))


__all__=['SCHEMA','ADMISSION_SCHEMA','TRUST_POLICY','IDENTITIES','ASSUMPTIONS','MEMORY_SCOPE','FP_POLICY',
         'canonical_sha256','validate_contract','validate_admission','inspect_stateful_bundle']

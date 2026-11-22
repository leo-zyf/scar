"""Typed transactional stateful profile for the shared final-entry capsule.

The original compact export is admission authority for the existing ABI and
guarded region. Its retained region-level selection never supplies final-entry
cost authority: this profile currently permits unmeasured validation only.
"""
from hashlib import sha256
from pathlib import Path

from .compiled_entry import artifact_reference,read_artifact,run_build
from .compiled_deployment import selected_rows
from .native_loop import _file_hash_and_header
from .stateful_contract import inspect_stateful_bundle,SCHEMA,ADMISSION_SCHEMA

PROFILE='COMPILED_STATEFUL_REGIONS_V1'


def _need(condition,message):
    if not condition:raise ValueError(message)


def runtime_sources():
    root=Path(__file__).resolve().parents[1]
    names=('entrypoint.py','__main__.py','runtime/compiled_stateful_capsule.py',
        'runtime/compiled_loop_capsule.py','runtime/compiled_entry.py','runtime/compiled_deployment.py',
        'runtime/entry_runner.py','runtime/native_stateful.py','runtime/stateful_contract.py',
        'runtime/stateful_guard.py','runtime/native_loop.py','runtime/native_temporal_invariant.py',
        'runtime/loop_guard.py','runtime/loop_contract.py',
        'runtime/array_copy_guard.py','trace/code_snapshot.py')
    return tuple(root/name for name in names)


def _original_evidence(admission,reference,role):
    expected=dict(role=role,path=reference['uri'],sha256=reference['sha256'],bytes=reference['size_bytes'])
    _need(expected in admission['evidence'],'original stateful '+role+' receipt differs')


def describe_pipeline(pipeline):
    path=Path(pipeline).resolve();path=path/'result.json'if path.is_dir()else path
    reference=artifact_reference(path,'scar.program-compiler-pipeline');program=read_artifact(reference)
    rows=selected_rows(program);_need(rows,'one or more admitted compact stateful regions required')
    records=[]
    for row in rows:
        _need(row.get('status')=='MEASURED_STATEFUL_BUNDLE_READY'
            and row.get('selected_kind')=='ORDERED_STATEFUL_REGION'
            and row.get('bundle_export') and row.get('native_artifact') and row.get('selection')
            and row.get('proof',{}).get('schema')=='scar.stateful-validation-candidate',
            'only original exported transactional stateful ABI1 regions are supported')
        exported=read_artifact(row['bundle_export']);directory=Path(row['bundle']).resolve()
        _need(exported.get('schema')=='scar.stateful-runtime-export'and exported['bundle']==str(directory),
            'original stateful compact export path/schema differs')
        metadata=inspect_stateful_bundle(directory,expected_admission_sha256=exported['expected_admission_sha256'])
        contract=metadata['contract'];admission=metadata['admission'];native=contract['native']
        _need(all(exported.get(key)==value for key,value in native['ids'].items())
            and exported['library_sha256']==native['library_sha256']
            and exported['runtime_contract_sha256']==admission['runtime_contract_sha256'],
            'stateful export/ABI identities differ')
        _original_evidence(admission,row['native_artifact'],'original-native-manifest')
        _original_evidence(admission,row['selection'],'original-measured-selection')
        shapes={spec['id']:spec['shape']for spec in contract['inputs']if spec['kind']=='array'}
        _need(shapes==row['input_shapes']==contract['selection']['input_shapes']
            and row['memory_budget_bytes']==contract['memory']['memory_budget_bytes']
            and row['resource_binding']['cpu_ids']==contract['cpu']['ids'],
            'stateful exact shape/budget/CPU specialization differs')
        path=directory/native['library'];digest,header=_file_hash_and_header(path)
        _need(digest==native['library_sha256']and header[:6]==b'\x7fELF\x02\x01',
            'original stateful ELF bytes differ')
        records.append(dict(id=row['id'],proof_kind='TRANSACTIONAL_STATEFUL_REGION',
            proof=row['proof'],schedule=row['schedule'],native_artifact=row['native_artifact'],
            original_selection=row['selection'],bundle=str(directory),bundle_export=row['bundle_export'],
            admission=artifact_reference(directory/'admission.json',ADMISSION_SCHEMA),
            runtime=artifact_reference(directory/'runtime.json',SCHEMA),
            native=dict(path=str(path),sha256=digest,bytes=path.stat().st_size),
            input_shapes=shapes,memory_budget_bytes=row['memory_budget_bytes'],
            external_assumptions=contract['external_assumptions'],
            floating_environment_policy=contract['floating_environment_policy'],
            transaction_boundary='ONE_ORIGINAL_METHOD_PRIVATE_SHADOW_WRITES_THEN_ALL_STATE_COMMIT'))
    return dict(profile_kind=PROFILE,pipeline=reference,rows=records)


def verify_profile(value):
    _need(value.get('profile_kind')==PROFILE,'registered compact stateful profile required')
    _need(describe_pipeline(value['pipeline']['uri'])==value,'compiled stateful profile changed')


def load_row(row):
    from .native_stateful import load
    return load(row['bundle'],expected_admission_sha256=row['admission']['sha256'])


def execute_profile(value,entry,argv):
    verify_profile(value)
    # This established runtime retains the ordinary method/class descriptor,
    # original callee/provider guards and whole-method precommit fallback.
    # The outer final capsule grants validation, never deployment permission.
    return run_build(value['pipeline']['uri'],entry,argv,exclusive_inputs=True,measure_candidate=True)


def execution_statistics(execution):
    if execution.get('schema')=='scar.entry-runner-execution':return dict(execution['statistics'])
    native=fallback=0;reasons={}
    for attachment in execution.get('attachments',()):
        for stats in attachment.get('statistics',()):
            native+=stats['native_calls'];fallback+=stats['fallback_calls']
            for reason,count in stats['reasons'].items():reasons[reason]=reasons.get(reason,0)+count
    return dict(native_calls=native,fallback_calls=fallback,reasons=reasons)


def main(argv=None):
    from .compiled_loop_capsule import _main
    return _main(argv,profile_kind=PROFILE)


if __name__=='__main__':raise SystemExit(main())

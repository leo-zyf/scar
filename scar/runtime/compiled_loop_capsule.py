"""Existing compact DIRECT/COMPOSITE fixed-loop profile for final delivery."""
import argparse
from hashlib import sha256
from pathlib import Path

from scar.runtime.compiled_entry import artifact_reference, read_artifact, run_build
from .compiled_deployment import selected_rows
from .native_loop import _small_raw,_json,_file_hash_and_header
from .native_temporal_invariant import validate_contract,SCHEMA,ADMISSION,TRUST

PROFILE='COMPILED_FIXED_COUNT_LOOPS_V1'


def _need(value,message):
    if not value:raise ValueError(message)


def runtime_sources():
    root=Path(__file__).resolve().parents[1]
    names=('entrypoint.py','__main__.py','runtime/compiled_loop_capsule.py','runtime/compiled_entry.py','runtime/compiled_deployment.py',
        'runtime/entry_runner.py','runtime/native_temporal_invariant.py','runtime/native_loop.py',
        'runtime/loop_guard.py','runtime/loop_contract.py','runtime/array_copy_guard.py','trace/code_snapshot.py')
    return tuple(root/name for name in names)


def describe_pipeline(pipeline):
    path=Path(pipeline).resolve();path=path/'result.json'if path.is_dir()else path
    reference=artifact_reference(path,'scar.program-compiler-pipeline');program=read_artifact(reference)
    rows=selected_rows(program);_need(rows,'one or more existing model-selected loops required')
    records=[]
    for row in rows:
        _need(row.get('selected_kind')=='REGISTER_TEMPORAL_FUSION'and row.get('bundle_export')
            and row['proof']['schema']in('scar.invariant-loop-proof','scar.composite-invariant-loop-proof'),
            'only compact original DIRECT/COMPOSITE fixed-loop artifacts are supported')
        exported=read_artifact(row['bundle_export']);directory=Path(row['bundle']).resolve()
        _need(exported['bundle']==str(directory),'retained compact bundle path differs')
        raw=_small_raw(directory/'admission.json');admission=_json(raw)
        _need(sha256(raw).hexdigest()==exported['expected_admission_sha256']
            and admission.get('schema')==ADMISSION and admission.get('schema_version')==1
            and admission['trust_policy']==TRUST,'original compact export admission required')
        raw=_small_raw(directory/'runtime.json');contract=validate_contract(_json(raw))
        _need(sha256(raw).hexdigest()==admission['runtime_sha256']==exported['runtime_sha256']
            and contract['native']['library_sha256']==admission['library_sha256'],'compact runtime/library identities differ')
        composite=contract.get('proof_kind','DIRECT')=='COMPOSITE'
        _need(composite==(row['proof']['schema']=='scar.composite-invariant-loop-proof')
            and contract['source_proof']==row['proof'],'original proof kind/source differs')
        schedule=contract['resource_schedule']['receipt']
        _need(schedule==dict(path=row['schedule']['uri'],sha256=row['schedule']['sha256'],bytes=row['schedule']['size_bytes']),
            'original selected schedule differs')
        _need(contract['trip_count']==row['trip_count']
            and {name:spec['shape']for name,spec in contract['arrays'].items()}==row['input_shapes']
            and contract['memory_budget_bytes']<=row['memory_budget_bytes'],'original shape/count/budget specialization differs')
        native=directory/contract['native']['library'];digest,header=_file_hash_and_header(native)
        _need(digest==contract['native']['library_sha256']and header[:6]==b'\x7fELF\x02\x01','original native ELF bytes differ')
        records.append(dict(id=row['id'],proof_kind='COMPOSITE'if composite else'DIRECT',proof=row['proof'],schedule=row['schedule'],
            bundle=str(directory),bundle_export=row['bundle_export'],admission=artifact_reference(directory/'admission.json',ADMISSION),
            runtime=artifact_reference(directory/'runtime.json',SCHEMA),native=dict(path=str(native),sha256=digest,bytes=native.stat().st_size),
            input_shapes=row['input_shapes'],trip_count=row['trip_count'],memory_budget_bytes=row['memory_budget_bytes']))
    return dict(profile_kind=PROFILE,pipeline=reference,rows=records)


def verify_profile(value):
    _need(value.get('profile_kind')==PROFILE,'registered compact loop profile required')
    _need(describe_pipeline(value['pipeline']['uri'])==value,'compiled pipeline/bundle profile changed')


def load_row(row):
    from .native_temporal_invariant import load
    return load(row['bundle'],expected_admission_sha256=row['admission']['sha256'])


def execute_profile(value,entry,argv):
    # The final outer gate grants only explicit, unmeasured validation. The
    # established runner retains original callables, guard and whole-loop fallback.
    verify_profile(value)
    return run_build(value['pipeline']['uri'],entry,argv,exclusive_inputs=True,measure_candidate=True)


def execution_statistics(execution):
    if execution.get('schema')=='scar.entry-runner-execution':return dict(execution['statistics'])
    native=fallback=0;reasons={}
    for attachment in execution.get('attachments',()):
        for stats in attachment.get('statistics',()):
            native+=stats['native_calls'];fallback+=stats['fallback_calls']
            for reason,count in stats['reasons'].items():reasons[reason]=reasons.get(reason,0)+count
    return dict(native_calls=native,fallback_calls=fallback,reasons=reasons)


def _main(argv=None,*,profile_kind=PROFILE):
    from scar.runtime.compiled_entry import run_final_entry
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capsule',required=True);parser.add_argument('--slot',required=True)
    parser.add_argument('--payload-sha256',required=True);parser.add_argument('--slot-sha256',required=True)
    parser.add_argument('--report',required=True);parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--exclusive-inputs',action='store_true');parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv);command=args.command[1:]if args.command[:1]==['--']else args.command
    if not command:parser.error('unchanged Python entry required after --')
    run_final_entry(command[0],command[1:],capsule=args.capsule,slot=args.slot,
        expected_payload_sha256=args.payload_sha256,expected_slot_sha256=args.slot_sha256,report=args.report,
        profile_kind=profile_kind,validation_only=args.validate_only,exclusive_inputs=args.exclusive_inputs)
    return 0


def main(argv=None):return _main(argv,profile_kind=PROFILE)


if __name__=='__main__':raise SystemExit(main())

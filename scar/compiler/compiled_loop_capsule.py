"""Adapt an existing selected loop pipeline to unmeasured final delivery.

This exports no mathematical code and no profitability approval. Existing
original proof/schedule/export identities and runtime guards remain intact.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from scar.runtime.compiled_entry import read_artifact,artifact_reference
from scar.runtime.final_entry_capsule import PAYLOAD,PROFILE,TRUST,runtime_identity,_profile
from scar.runtime.caller_pipeline_capsule import pack_slot,SLOT_BYTES,TRIAL,SLOT
from scar.runtime.caller_pipeline_deployment import entry_context,environment_sha256,provider_identity
from scar.compiler.caller_pipeline_capsule import _slot


def _sources(value):
    if type(value)is dict:
        if {'path','sha256','code_key','qualname'}<=set(value):yield value
        else:
            for item in value.values():yield from _sources(item)
    elif type(value)is list:
        for item in value:yield from _sources(item)


def prepare_profile_capsule(pipeline,out,*,profile_kind,entry,argv=(),source_paths=(),export_schema):
    """Seal typed compact loop artifacts; export one explicitly unknown slot."""
    from scar.runtime.entry_runner import _arguments
    entry,argv=_arguments(entry,argv);out=Path(out).resolve()
    if out.exists():raise ValueError('fresh final loop capsule required')
    adapter=_profile(profile_kind);profile=adapter.describe_pipeline(pipeline);versions={};parents=[]
    for row in profile['rows']:
        artifact=adapter.load_row(row)
        for source in _sources(artifact.contract):
            path=Path(source['path']).resolve();digest=sha256(path.read_bytes()).hexdigest()
            if digest!=source['sha256']or versions.setdefault(path,digest)!=digest:
                raise ValueError('original loop/callee source version differs')
        for reference in (row['proof'],row['schedule']):read_artifact(reference);parents.append(reference)
    paths={entry,*(Path(path).resolve()for path in source_paths),*versions}
    ledger=[artifact_reference(path,'scar.final-entry-program-source')for path in sorted(paths)]
    payload=dict(schema=PAYLOAD,schema_version=1,trust_policy=TRUST,profile_kind=profile_kind,artifact_profile=profile,
        capsule_directory=str(out),slot_bytes=SLOT_BYTES,entry_context=entry_context(entry,argv),
        environment_sha256=environment_sha256(),source_refs=ledger,provider_identity=provider_identity(),
        runtime_identity=runtime_identity(profile_kind),validator_implementation=artifact_reference(__file__,'scar.validation-raw'),
        policy=dict(warmup=1,repetitions=3,minimum_gain=.05,frozen_before_measurement=True),
        reporting=dict(required=True,allowed_root=str(out.parent)),
        cost_status='UNMEASURED',deployment_authorized=False,original_performance_evidence_used=False,
        numeric_cost_wire_schema=SLOT,original_guard_effect_shape_budget_conditions_reused=True)
    raw=(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();digest=sha256(raw).hexdigest()
    unknown=_slot(payload,digest,TRIAL)
    out.mkdir(parents=True);(out/'payload.json').write_bytes(raw);(out/'validation.slot').write_bytes(pack_slot(unknown))
    result=dict(schema=export_schema,schema_version=1,
        payload=artifact_reference(out/'payload.json',PAYLOAD),slot=artifact_reference(out/'validation.slot',SLOT),
        original_pipeline=profile['pipeline'],original_proof_schedule_parents=parents,
        original_cpp_bytes_reused=True,new_math_backend=False,performance_measured=False,
        release_exported=False,deployment_authorized=False,cost_status='UNMEASURED')
    (out/'export.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return result


def prepare_loop_capsule(pipeline,out,*,entry,argv=(),source_paths=()):
    return prepare_profile_capsule(pipeline,out,profile_kind=PROFILE,entry=entry,argv=argv,
        source_paths=source_paths,export_schema='scar.compiled-loop-final-capsule-export')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--source',action='append',default=[]);parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv);command=args.command[1:]if args.command[:1]==['--']else args.command
    if not command:parser.error('unchanged Python entry required after --')
    result=prepare_loop_capsule(args.pipeline,args.out,entry=command[0],argv=command[1:],source_paths=args.source)
    print(json.dumps(result,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())

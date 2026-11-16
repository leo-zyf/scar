"""Validate final caller delivery using the source set of an admitted build.

The wrapper runs at build time. The measured runtime remains the unchanged
compact capsule module, so this convenience command adds no deployment work.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path


def validate_caller_delivery(program,bundle,out,argv,*,expected_admission_sha256,
        warmup=1,repetitions=3,minimum_gain=.05):
    from scar.analysis.build import BuildSession
    from scar.compiler.caller_pipeline_capsule import prepare_capsule,validate_capsule,export_cost_slot
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh final caller delivery directory required')
    with BuildSession(program)as session:
        if session.report.get('backend')!='F4':raise ValueError('an admitted original F4 build required')
        source=[]
        for unit,row in session.view.catalog.fact_store.rows.items():
            path=Path(row['path']).resolve();model=session.view.catalog.fact_store.model_units[unit]
            if sha256(path.read_bytes()).hexdigest()!=model['raw_sha256']:
                raise ValueError('declared source differs from retained compiler input: '+str(path))
            source.append(path)
        entry=session.report['entry']
    out.mkdir(parents=True)
    prepared=prepare_capsule(bundle,entry,argv,out/'capsule',
        expected_admission_sha256=expected_admission_sha256,source_paths=source,
        warmup=warmup,repetitions=repetitions,minimum_gain=minimum_gain)
    validated=validate_capsule(out/'capsule',out/'validation',
        expected_payload_sha256=prepared['payload']['sha256'],
        expected_trial_slot_sha256=prepared['trial_slot']['sha256'])
    decision=export_cost_slot(out/'capsule',validated['validation']['uri'],
        expected_payload_sha256=prepared['payload']['sha256'],
        expected_validation_sha256=validated['validation']['sha256'])
    result=dict(schema='scar.build-final-caller-delivery',schema_version=1,program=str(Path(program).resolve()),
        source_set_from_original_catalog=True,source_files=len(source),prepared=prepared,
        validation=validated['validation'],decision=decision,physical_optimality_proven=False)
    (out/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',required=True);parser.add_argument('--bundle',required=True)
    parser.add_argument('--admission-sha256',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--warmup',type=int,default=1);parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--minimum-gain',type=float,default=.05);parser.add_argument('args',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv);command=args.args[1:]if args.args[:1]==['--']else args.args
    result=validate_caller_delivery(args.build,args.bundle,args.out,command,
        expected_admission_sha256=args.admission_sha256,warmup=args.warmup,
        repetitions=args.repetitions,minimum_gain=args.minimum_gain)
    print(json.dumps(dict(decision=result['decision']['decision'],source_files=result['source_files'])));return 0


if __name__=='__main__':raise SystemExit(main())

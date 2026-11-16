"""Seal an existing transactional stateful bundle for unmeasured delivery."""
import argparse
import json

from scar.runtime.compiled_stateful_capsule import PROFILE
from .compiled_loop_capsule import prepare_profile_capsule


def prepare_stateful_capsule(pipeline,out,*,entry,argv=(),source_paths=()):
    return prepare_profile_capsule(pipeline,out,profile_kind=PROFILE,entry=entry,argv=argv,
        source_paths=source_paths,export_schema='scar.compiled-stateful-final-capsule-export')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--source',action='append',default=[]);parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv);command=args.command[1:]if args.command[:1]==['--']else args.command
    if not command:parser.error('unchanged Python entry required after --')
    result=prepare_stateful_capsule(args.pipeline,args.out,entry=command[0],argv=command[1:],source_paths=args.source)
    print(json.dumps(result,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())

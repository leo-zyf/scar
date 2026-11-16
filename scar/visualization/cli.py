"""Export retained SCAR graphs to an offline interactive canvas."""
import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import _bytes, _json, read_artifact
from .adapters import project_schedule
from .bundle import write_view_bundle


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--build',help='first-stage ProgramBuild directory or manifest')
    inputs.add_argument('--projection',help='retained bounded ProgramView projection')
    inputs.add_argument('--schedule',help='retained v2/v3 task graph JSON')
    inputs.add_argument('--time-schedule',help='source-replayed time preparation receipt')
    parser.add_argument('--out',required=True)
    parser.add_argument('--hardware',help='retained target graph directory; inferred for a ProgramBuild')
    parser.add_argument('--title')
    parser.add_argument('--max-nodes',type=int,default=400)
    parser.add_argument('--max-parts',type=int,default=32)
    parser.add_argument('--depth',type=int,default=3)
    parser.add_argument('--max-edges',type=int,default=1600)
    parser.add_argument('--max-seconds',type=float,default=60)
    args=parser.parse_args(argv);hardware=args.hardware
    if args.build:
        from .project import project_build
        path=Path(args.build).resolve();path=path.parent if path.is_file() else path
        projection=project_build(path,max_nodes_per_part=args.max_nodes,max_parts=args.max_parts,
            max_depth=args.depth,max_edges_per_part=args.max_edges,max_seconds=args.max_seconds)
        if hardware is None:
            target=_json(_bytes(path/'manifest.json')).get('target')
            if target is not None:hardware=Path(target['path']).parent
    elif args.projection:
        projection=_json(_bytes(args.projection))
    elif args.schedule:
        projection=project_schedule(_json(_bytes(args.schedule)))
    else:
        from scar.compiler.time_schedule import read_time_schedule
        receipt=read_time_schedule(args.time_schedule)
        projection=project_schedule(read_artifact(receipt['schedule']))
        if hardware is None and receipt.get('target') is not None:
            hardware=Path(receipt['target']['uri']).parent
    result=write_view_bundle(projection,args.out,title=args.title,hardware=hardware)
    print(json.dumps(dict(entry=result['entry'],parts=result['parts'],
        graph_data_bytes=result['graph_data_bytes'],resource_stats=result['resource_stats'],
        coverage=result['coverage'],ir_modified=False,network_required_at_view_time=False),sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())

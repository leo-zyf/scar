"""Compose dependency review, forward CPU costs and selected C++ lowering.

Consumes a retained plan-build, sharing its original region and hardware.
Missing proofs/costs remain visible. Compiling does not deploy a replacement
or certify whole-program equivalence.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time


def compile_build(domains, out, *, calibrations=(), no_op_profiles=None, plan_only=False,
                  loop_trip_count=None,loop_inputs=None):
    from scar.analysis.build import _check
    from scar.analysis.stateful_dependencies import project_stateful_dependencies
    from scar.compiler.choose_stateful_cpu import choose_stateful_cpu
    from scar.compiler.lower_stateful_choices import lower_stateful_choices
    from scar.ir.execution_recipe import artifact_reference,read_artifact
    from scar.optimization.stateful_cpu_choices import _domain
    from scar.optimization.stateful_cpu_model import build_stateful_cpu_profiles
    domains=Path(domains).resolve();out=Path(out).resolve()
    if domains.is_dir():domains=domains/'result.json'
    if out.exists():raise ValueError('fresh compile-build output required')
    parent=artifact_reference(domains,'scar.program-optimization-domains');document=read_artifact(parent)
    if document.get('schema_version')!=1:raise ValueError('supported program planning domain required')
    read_artifact(document['program']);_check(document['hardware'])
    loop_ref=document.get('loop_dependencies')
    if loop_ref is not None:read_artifact(loop_ref)
    no_op_profiles={} if no_op_profiles is None else no_op_profiles
    identifiers={row['region_id'] for row in document['regions']}
    if set(no_op_profiles)-identifiers:raise ValueError('NO_OP profiles must name retained original regions')
    started=time.monotonic();out.mkdir(parents=True);rows=[]
    for index,original in enumerate(document['regions']):
        directory=out/('region-'+str(index));directory.mkdir()
        row=dict(region_id=original['region_id'],selected_for_use=False,validation='NOT_EXECUTED')
        projection=project_stateful_dependencies(original['region'],out=directory/'dependencies')
        row['dependencies']=artifact_reference(directory/'dependencies/dependencies.json',projection['schema'])
        if original['status']!='HARDWARE_DOMAIN_BUILT':
            row.update(status=original['status'],reason='input constraints required before hardware planning')
            rows.append(row);continue
        domain=_domain(original['domain'])
        if domain['region']['uri']!=original['region'] or projection['region_id']!=original['region_id']:
            raise ValueError('planning domain differs from retained original region')
        target=domain['target_snapshot'];hardware=document['hardware']
        if (target['uri'],target['sha256'],target['size_bytes'])!=(hardware['path'],hardware['sha256'],hardware['bytes']):
            raise ValueError('planning domain differs from the program hardware snapshot')
        profiles=build_stateful_cpu_profiles(domain,calibrations,out=directory/'costs')
        choice=choose_stateful_cpu(directory/'planning',domain=original['domain'],
            cost_profiles=profiles['cost_profiles'],
            no_op_profile=no_op_profiles.get(original['region_id'],profiles['no_op_profile']))
        lowered=lower_stateful_choices(choice['choice'],directory/'lowering',plan_only=plan_only)
        gaps=Counter();rejected=Counter()
        for model in profiles['reports']:
            report=json.loads(Path(model['model']).read_bytes())
            for issue in report['unresolved']:
                gaps[issue['component']]+=1
                if issue.get('rejected_estimates'):rejected[issue['component']]+=1
        row.update(status=lowered['status'],choice=choice['choice'],lowering=str(directory/'lowering/lowering.json'),
            native_directory=lowered['native_directory'],candidate_count=choice['candidate_count'],
            complete_cost_profiles=len(profiles['cost_profiles']),cost_domain_complete=profiles['complete_domain'],
            unresolved_components=dict(gaps),rejected_components=dict(rejected),
            caller_loop_proof='NOT_ESTABLISHED_BY_PER_CALL_TRANSFER',
            automatic_caller_replacement_authorized=False)
        rows.append(row)
    from scar.compiler.invariant_build import compile_invariant_build
    loops=compile_invariant_build(document,out/'loops',loop_trip_count=loop_trip_count,
        loop_inputs=loop_inputs,plan_only=plan_only)
    read_artifact(parent);read_artifact(document['program']);_check(document['hardware'])
    report=dict(schema='scar.program-compiler-pipeline',schema_version=1,parent=parent,
        program=document['program'],hardware=document['hardware'],loop_dependencies=loop_ref,
        regions=rows,invariant_loops=loops,source_rejections=document['rejections'],
        scope='GUARDED_STATEFUL_REGIONS_WITH_EXPLICIT_CALLER_AND_COST_FRONTIERS',
        source_graph_unrolled=False,target_variants_ranked_by_measurement=False,
        target_program_executed=False,whole_program_compiled=False,three_stages_complete=False,
        selected_for_use=False,seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domains',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--calibration',action='append',default=[])
    parser.add_argument('--no-op-profiles',help='JSON mapping region IDs to same-boundary original-call profiles')
    loops=parser.add_mutually_exclusive_group()
    loops.add_argument('--loop-trip-count',type=int,help='declared common trip count for retained invariant loops')
    loops.add_argument('--loop-inputs',help='JSON mapping original loop IDs to input_shapes/trip_count/budget constraints')
    parser.add_argument('--plan-only',action='store_true');args=parser.parse_args(argv)
    result=compile_build(args.domains,args.out,calibrations=args.calibration,
        no_op_profiles=None if args.no_op_profiles is None else json.loads(Path(args.no_op_profiles).read_bytes()),
        plan_only=args.plan_only,loop_trip_count=args.loop_trip_count,
        loop_inputs=None if args.loop_inputs is None else json.loads(Path(args.loop_inputs).read_bytes()))
    print(json.dumps(dict(regions=len(result['regions']),invariant_loops=len(result['invariant_loops']),
        statuses=[r['status'] for r in [*result['regions'],*result['invariant_loops']]],
        result=str(Path(args.out)/'result.json'),seconds=result['seconds']),indent=2));return 0


if __name__=='__main__':raise SystemExit(main())

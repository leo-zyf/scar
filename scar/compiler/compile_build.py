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
                  loop_trip_count=None,loop_inputs=None,prepare_stateful_validation=False,stateful_cpu_threads=1,
                  loop_cpu_threads=None,loop_cpu_model=False,loop_calibrations=(),loop_no_op_profiles=None,
                  loop_max_threads=None,stateful_interval_plans=None):
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
    stateful_interval_plans={} if stateful_interval_plans is None else stateful_interval_plans
    if not isinstance(stateful_interval_plans,dict):raise ValueError('interval plans require a region-id to plan mapping')
    if any(not isinstance(value,(str,Path,dict)) for value in stateful_interval_plans.values()):
        raise ValueError('each region interval plan must be an original plan path or ArtifactRef')
    identifiers={row['region_id'] for row in document['regions']}
    if set(no_op_profiles)-identifiers:raise ValueError('NO_OP profiles must name retained original regions')
    if set(stateful_interval_plans)-identifiers:raise ValueError('interval plans must name retained original regions')
    if set(stateful_interval_plans)&set(no_op_profiles):
        raise ValueError('one region cannot consume both an interval plan and scalar NO_OP profiles')
    started=time.monotonic();out.mkdir(parents=True);rows=[]
    for index,original in enumerate(document['regions']):
        directory=out/('region-'+str(index));directory.mkdir()
        row=dict(region_id=original['region_id'],selected_for_use=False,validation='NOT_EXECUTED')
        graph=original.get('program_graph')
        if graph is not None:
            from scar.compiler.ordered_planning_graph import admit_ordered_planning_graph
            row['program_graph']=graph
            row['program_graph_admission']=admit_ordered_planning_graph(graph,
                program=document['program'],region=original['region'],out=directory/'graph-admission.json')
        projection=project_stateful_dependencies(original['region'],out=directory/'dependencies')
        row['dependencies']=artifact_reference(directory/'dependencies/dependencies.json',projection['schema'])
        if original['status']!='HARDWARE_DOMAIN_BUILT':
            if original['region_id'] in stateful_interval_plans:
                raise ValueError('interval-plan region needs its retained original hardware domain')
            row.update(status=original['status'],reason='input constraints required before hardware planning')
            rows.append(row);continue
        domain=_domain(original['domain'])
        if domain['region']['uri']!=original['region'] or projection['region_id']!=original['region_id']:
            raise ValueError('planning domain differs from retained original region')
        target=domain['target_snapshot'];hardware=document['hardware']
        if (target['uri'],target['sha256'],target['size_bytes'])!=(hardware['path'],hardware['sha256'],hardware['bytes']):
            raise ValueError('planning domain differs from the program hardware snapshot')
        interval_value=stateful_interval_plans.get(original['region_id'])
        if interval_value is not None:
            from scar.compiler.lower_stateful_bounds import lower_stateful_bounds
            from scar.compiler.stateful_domain_equivalence import retain_stateful_domain_equivalence
            if isinstance(interval_value,(str,Path)):
                interval_path=Path(interval_value).resolve()
                if interval_path.is_dir():interval_path/='result.json'
                interval_ref=artifact_reference(interval_path,'scar.stateful-interval-choice-plan')
            elif isinstance(interval_value,dict):
                interval_ref=interval_value;interval_path=Path(interval_ref['uri'])
                if interval_ref.get('schema')!='scar.stateful-interval-choice-plan':
                    raise ValueError('retained interval-choice plan ArtifactRef required')
            else:raise ValueError('interval plans must contain original plan paths or ArtifactRefs')
            interval=read_artifact(interval_ref)
            main_domain=artifact_reference(original['domain'],'scar.stateful-cpu-domain')
            equivalence=retain_stateful_domain_equivalence(interval['original_domain'],main_domain,
                directory/'domain-equivalence.json')
            # This existing lowering owns complete source/fact/interval replay.
            # Do not run scalar calibration/choice or an unranked preview here.
            lowered=lower_stateful_bounds(interval_path,directory/'interval-lowering',plan_only=plan_only)
            if lowered['upstream_plan']!=interval_ref or lowered['domain']!=interval['original_domain']:
                raise ValueError('verified interval lowering differs from the retained source plan/domain')
            read_artifact(interval_ref);bounds=read_artifact(interval['source_bounds_artifact'])
            row.update(status=lowered['status'],planning_route='VERIFIED_INTERVAL_PLAN',
                domain=main_domain,interval_source_domain=interval['original_domain'],
                interval_plan=interval_ref,interval_choice=interval['interval_choice'],
                interval_lowering=artifact_reference(directory/'interval-lowering/lowering.json',lowered['schema']),
                domain_equivalence=equivalence,lowering=str(directory/'interval-lowering/lowering.json'),
                native_directory=lowered['native_directory'],candidate_count=len(domain['candidates']),
                complete_cost_profiles=bounds['counts']['previously_complete_options']-1,
                cost_domain_complete=False,interval_choice_complete=interval['status']=='FINITE_MODEL_OPTIMAL',
                interval_status=interval['status'],selected_candidate_id=interval['selected_candidate_id'],
                remaining_cost_frontier=interval['remaining_frontier'],
                partial_options_strictly_dominated=interval['prior_partial_options_strictly_dominated'],
                scalar_calibration_and_choice_consumed=False,
                caller_loop_proof='NOT_ESTABLISHED_BY_PER_CALL_TRANSFER',
                automatic_caller_replacement_authorized=False)
            rows.append(row);continue
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
        if prepare_stateful_validation and not plan_only and lowered['native_directory'] is None:
            from scar.compiler.stateful_candidate import prepare_stateful_candidate
            candidate=prepare_stateful_candidate(document['program']['uri'],domain,directory/'validation-candidate',
                cpu_threads=stateful_cpu_threads)
            row['validation_candidate']=artifact_reference(directory/'validation-candidate/candidate.json',candidate['schema'])
        rows.append(row)
    from scar.compiler.invariant_build import compile_invariant_build
    from scar.compiler.composite_invariant_build import compile_composite_invariant_build
    original_ids={row['id'] for row in document.get('invariant_loops',())}
    composite_ids={row['id'] for row in document.get('composite_invariant_loops',())}
    if loop_inputs is not None and set(loop_inputs)-(original_ids|composite_ids):
        raise ValueError('loop inputs name an unknown original or composite loop')
    loop_no_op_profiles={} if loop_no_op_profiles is None else loop_no_op_profiles
    if set(loop_no_op_profiles)-(original_ids|composite_ids):
        raise ValueError('loop original costs name an unknown original or composite loop')
    original_inputs=None if loop_inputs is None else {key:value for key,value in loop_inputs.items() if key in original_ids}
    composite_inputs=None if loop_inputs is None else {key:value for key,value in loop_inputs.items() if key in composite_ids}
    loops=compile_invariant_build(document,out/'loops',loop_trip_count=loop_trip_count,
        loop_inputs=original_inputs,plan_only=plan_only,loop_cpu_threads=loop_cpu_threads,
        loop_cpu_model=loop_cpu_model,loop_calibrations=loop_calibrations,
        loop_no_op_profiles={key:value for key,value in loop_no_op_profiles.items() if key in original_ids},
        loop_max_threads=loop_max_threads)
    composite_loops=compile_composite_invariant_build(document,out/'composite-loops',loop_trip_count=loop_trip_count,
        loop_inputs=composite_inputs,plan_only=plan_only,loop_cpu_threads=loop_cpu_threads,
        cpu_cost_model=loop_cpu_model,loop_calibrations=loop_calibrations,
        loop_no_op_profiles={key:value for key,value in loop_no_op_profiles.items() if key in composite_ids},
        loop_max_threads=loop_max_threads)
    read_artifact(parent);read_artifact(document['program']);_check(document['hardware'])
    report=dict(schema='scar.program-compiler-pipeline',schema_version=1,parent=parent,
        program=document['program'],hardware=document['hardware'],loop_dependencies=loop_ref,
        regions=rows,invariant_loops=loops,source_rejections=document['rejections'],
        source_rejection_views=dict(
            invariant_loops=dict(artifact=parent,locator='#/invariant_loop_rejections',
                count=len(document.get('invariant_loop_rejections',[]))),
            composite_loops=dict(artifact=parent,locator='#/composite_rejections',
                count=len(document.get('composite_rejections',[])))),
        composite_invariant_loops=composite_loops,
        scope='ORIGINAL_F4_STATEFUL_AND_PURE_LOOP_REGIONS_WITH_SCOPED_CALLER_COST_AND_RUNTIME_CONTRACTS',
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
    parser.add_argument('--stateful-interval-plans',help='JSON mapping exact region IDs to retained interval-choice plan paths or ArtifactRefs')
    loops=parser.add_mutually_exclusive_group()
    loops.add_argument('--loop-trip-count',type=int,help='declared common trip count for retained invariant loops')
    loops.add_argument('--loop-inputs',help='JSON mapping original loop IDs to input_shapes/trip_count/budget constraints')
    parser.add_argument('--prepare-stateful-validation',action='store_true',help='prepare one original-F4-proven stateful candidate without changing cost selection')
    parser.add_argument('--stateful-cpu-threads',type=int,default=1,help='explicit unranked worker count for stateful validation preview')
    cpu=parser.add_mutually_exclusive_group()
    cpu.add_argument('--loop-cpu-threads',type=int,help='explicit hardware-checked OpenMP team preview; not a claim of best worker count')
    cpu.add_argument('--loop-cpu-model',action='store_true',help='select only from a complete sourced forward CPU component cost domain')
    parser.add_argument('--loop-calibration',action='append',default=[],help='independent fixed-family temporal resource observations or projected component facts')
    parser.add_argument('--loop-no-op-profiles',help='JSON mapping original loop IDs to retained same-boundary original-call cost profiles')
    parser.add_argument('--loop-max-threads',type=int,help='explicit maximum team size for the cost model; all representative masks within that limit remain visible')
    parser.add_argument('--plan-only',action='store_true');args=parser.parse_args(argv)
    result=compile_build(args.domains,args.out,calibrations=args.calibration,
        no_op_profiles=None if args.no_op_profiles is None else json.loads(Path(args.no_op_profiles).read_bytes()),
        plan_only=args.plan_only,loop_trip_count=args.loop_trip_count,
        loop_inputs=None if args.loop_inputs is None else json.loads(Path(args.loop_inputs).read_bytes()),
        prepare_stateful_validation=args.prepare_stateful_validation,stateful_cpu_threads=args.stateful_cpu_threads,
        loop_cpu_threads=args.loop_cpu_threads,loop_cpu_model=args.loop_cpu_model,
        loop_calibrations=args.loop_calibration,
        loop_no_op_profiles=None if args.loop_no_op_profiles is None else json.loads(Path(args.loop_no_op_profiles).read_bytes()),
        loop_max_threads=args.loop_max_threads,
        stateful_interval_plans=None if args.stateful_interval_plans is None else json.loads(Path(args.stateful_interval_plans).read_bytes()))
    print(json.dumps(dict(regions=len(result['regions']),invariant_loops=len(result['invariant_loops']),
        composite_invariant_loops=len(result['composite_invariant_loops']),
        statuses=[r['status'] for r in [*result['regions'],*result['invariant_loops'],*result['composite_invariant_loops']]],
        result=str(Path(args.out)/'result.json'),seconds=result['seconds']),indent=2));return 0


if __name__=='__main__':raise SystemExit(main())

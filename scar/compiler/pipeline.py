"""Compile retained Python region domains through the shared C++20 driver."""

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import fields
import json
from pathlib import Path
import time

from scar.analysis.source_decoded_bodies import reuse_decoded_source_bodies
from scar.ir.execution_recipe import artifact_reference, open_artifact, read_artifact

from .contracts import EntryEvidence


_ENTRY_RESULT_LINKS = (
    'cpu_native_array_route', 'cpu_native_array_route_cost', 'cpu_fixed_grid_cli_review',
    'cpu_fixed_grid_mapped_input_plan', 'cpu_fixed_grid_lifecycle_cost',
    'cpu_fixed_grid_input_route_review', 'cpu_fixed_grid_source_qualification', 'cpu_fixed_grid_forward',
)


def _publish_result(out, parent, document, started, **details):
    from scar.analysis.build import _check

    read_artifact(parent)
    read_artifact(document['program'])
    _check(document['hardware'])
    report = dict(schema='scar.program-compiler-pipeline', schema_version=1, parent=parent,
                  program=document['program'], hardware=document['hardware'],
                  selected_for_use=False, three_stages_complete=False,
                  seconds=time.monotonic() - started, **details)
    if (admission := document.get('source_admission')) is not None:
        report['source_admission'] = admission
    (out / 'result.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report


def _compile_loops(document, out, *, trip_count, inputs, plan_only, cpu_threads,
                   cpu_model, calibrations, no_op_profiles, max_threads):
    from scar.analysis.build import BuildSession, _check
    from .backend.cpp20 import deliver
    from .middle.python import schedule_loops

    has_loops = (document.get('invariant_loops') or document.get('composite_invariant_loops')
                 or document.get('array_loops') or inputs or no_op_profiles)
    admission = document.get('source_admission')
    if admission is None and not has_loops:
        return [], [], []
    program, hardware = document['program'], document['hardware']
    with BuildSession(Path(program['uri']).parent) as session:
        if admission is not None:
            from scar.analysis.build_admission import verify_build_admission

            verify_build_admission(admission, program=program, hardware=hardware,
                                   source_filter=document.get('source_filter'), _session=session)
        if not has_loops:
            return [], [], []
        direct, composite, arrays, work = schedule_loops(
            document, out, session.view, trip_count=trip_count, input_specs=inputs,
            cpu_threads=cpu_threads, cpu_model=cpu_model, calibrations=calibrations,
            original_costs=no_op_profiles, max_threads=max_threads,
        )
        for row, loop in work:
            row.update(deliver(loop, catalog=session.view, plan_only=plan_only))
    read_artifact(program)
    _check(hardware)
    return direct, composite, arrays


@reuse_decoded_source_bodies()
def compile_build(domains, out, *, calibrations=(), no_op_profiles=None, plan_only=False,
                  loop_trip_count=None, loop_inputs=None, prepare_stateful_validation=False, stateful_cpu_threads=1,
                  loop_cpu_threads=None, loop_cpu_model=False, loop_calibrations=(), loop_no_op_profiles=None,
                  loop_max_threads=None, stateful_interval_plans=None, entry_evidence: EntryEvidence | None = None,
                  verify_entry=False, enforce_cpu_capacity=False):
    from scar.analysis.build import _check

    domains=Path(domains).resolve();out=Path(out).resolve()
    if domains.is_dir():domains=domains/'result.json'
    if out.exists():raise ValueError('fresh compile-build output required')
    started=time.monotonic()
    parent, document = open_artifact(domains, 'scar.program-optimization-domains')
    evidence = EntryEvidence() if entry_evidence is None else entry_evidence
    if enforce_cpu_capacity and (not verify_entry or plan_only or 'whole_entry' not in document):
        raise ValueError('CPU capacity enforcement requires a whole-entry process trial')
    if 'whole_entry' not in document and evidence != EntryEvidence(costs=evidence.costs):
        raise ValueError('CPU entry evidence requires a whole-entry plan')
    if evidence.array_route_cost is not None and evidence.array_route is None:
        raise ValueError('CPU native array route costs require the exact C++ route package')
    version=document.get('schema_version')
    if version not in (1,2):raise ValueError('supported program planning domain required')
    read_artifact(document['program']);_check(document['hardware'])
    source_admission=document.get('source_admission')
    if version==2 and source_admission is None:
        raise ValueError('program planning domain is missing original graph admission')
    loop_ref=document.get('loop_dependencies')
    if loop_ref is not None:read_artifact(loop_ref)
    if 'whole_entry' in document:
        if source_admission is None:
            raise ValueError('whole-entry lowering requires a replayed original graph admission')
        if (calibrations or loop_calibrations or prepare_stateful_validation or loop_cpu_model
                or stateful_cpu_threads!=1 or any(value is not None for value in (no_op_profiles,
                loop_trip_count,loop_inputs,loop_cpu_threads,loop_no_op_profiles,loop_max_threads,stateful_interval_plans))):
            raise ValueError('whole-entry request already fixes the scope and declared implementation evidence')
        if plan_only and verify_entry:
            raise ValueError('process validation requires native emission')
        from scar.analysis.replay_transaction import replay_transaction
        from scar.compiler.whole_entry_pipeline import replay_whole_entry
        out.mkdir(parents=True)
        paired = (evidence.needs_paired_replay
                  or 'cpu_paired_observation' in document['whole_entry']['scope'])
        with replay_transaction() if paired else nullcontext():
            whole = replay_whole_entry(document, out, evidence=evidence, plan_only=plan_only,
                                       verify_entry=verify_entry, enforce_cpu_capacity=enforce_cpu_capacity)
        return _publish_result(out, parent, document, started,
            whole_entry=whole['reference'],regions=[],invariant_loops=[],composite_invariant_loops=[],array_loops=[],
            whole_entry_status=whole['selection']['status'],whole_entry_action=whole['delivery']['final_action'],
            scope='REPLAYED_WHOLE_ENTRY_OPTION_DOMAIN',
            whole_program_compiled=whole['native_compiled'],target_program_executed=whole['target_executed'],
            **{key: whole[key] for key in _ENTRY_RESULT_LINKS if key in whole})
    if evidence.costs is not None or verify_entry:
        raise ValueError('entry cost evidence and process validation require a whole-entry domain')

    from scar.analysis.stateful_dependencies import project_stateful_dependencies
    from scar.compiler.choose_stateful_cpu import choose_stateful_cpu
    from scar.compiler.selected_stateful_lowering import lower_stateful_choices
    from scar.optimization.stateful_cpu_model import _domain, build_stateful_cpu_profiles

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
    out.mkdir(parents=True)
    loops, composite_loops, array_loops = _compile_loops(
        document, out, trip_count=loop_trip_count, inputs=loop_inputs,
        plan_only=plan_only, cpu_threads=loop_cpu_threads, cpu_model=loop_cpu_model,
        calibrations=loop_calibrations, no_op_profiles=loop_no_op_profiles,
        max_threads=loop_max_threads,
    )
    rows=[]
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
            from scar.compiler.selected_stateful_lowering import lower_stateful_bounds
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
    return _publish_result(out, parent, document, started, loop_dependencies=loop_ref,
        regions=rows,invariant_loops=loops,source_rejections=document['rejections'],
        source_rejection_views=dict(
            invariant_loops=dict(artifact=parent,locator='#/invariant_loop_rejections',
                count=len(document.get('invariant_loop_rejections',[]))),
            composite_loops=dict(artifact=parent,locator='#/composite_rejections',
                count=len(document.get('composite_rejections',[]))),
            array_loops=dict(artifact=parent,locator='#/array_loop_rejections',
                count=len(document.get('array_loop_rejections',[])))),
        composite_invariant_loops=composite_loops,array_loops=array_loops,
        scope='ORIGINAL_F4_STATEFUL_AND_PURE_LOOP_REGIONS_WITH_SCOPED_CALLER_COST_AND_RUNTIME_CONTRACTS',
        source_graph_unrolled=False,target_variants_ranked_by_measurement=False,
        target_program_executed=False,whole_program_compiled=False)


def compile_domains_main(argv=None):
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
    parser.add_argument('--verify-entry',action='store_true',
        help='validate the selected entry or the eligible source-derived device and CPU nominations')
    parser.add_argument('--enforce-cpu-capacity',action='store_true',
        help='require a proved native CPU address-space contract for the entry trial')
    parser.add_argument('--cpu-entry-forward',dest='forward_cpu',action='store_true',
        help='replay hardware-aware component costs of an independently nominated CPU entry')
    parser.add_argument('--cpu-launch-probe',dest='launch_probes',nargs=1,default=(),
        help='attach an independent CPU process proxy observation to the forward report')
    for flag, role, help_text in (
        ('--entry-costs', 'costs', 'original whole-entry domain with late-bound model cost facts'),
        ('--cpu-team-probe', 'team_probe', 'independent component observations for the retained CPU team'),
        ('--cpu-native-phase-cost', 'native_phase_cost', 'same-provider observations of native phases'),
        ('--cpu-python-import-cost', 'python_import_cost', 'original CPython import observations'),
        ('--cpu-paired-evidence', 'paired_observation', 'checked original and native CPU observations'),
        ('--cpu-import-bindings', 'import_bindings', 'observed import selectors for the F4 initializer graph'),
        ('--cpu-import-sources', 'import_sources', 'exact source initializers from CPU import observations'),
        ('--cpu-initializer-observation', 'initializer_observation', 'observed module code entry and exit'),
        ('--cpu-native-array-route', 'array_route', 'scoped C++20 array task route'),
        ('--cpu-native-array-route-cost', 'array_route_cost', 'replayed timeline for the native array route'),
        ('--cpu-fixed-grid-cli-evidence', 'fixed_grid_cli', 'source-compatible fixed-grid C++ entry'),
        ('--cpu-fixed-grid-mapped-input-plan', 'fixed_grid_mapped_input', 'scoped copy versus mapping decision'),
        ('--cpu-fixed-grid-lifecycle-cost', 'fixed_grid_lifecycle_cost', 'fixed-grid lifecycle evidence'),
        ('--cpu-fixed-grid-input-route-review', 'fixed_grid_input_review', 'verified native input route'),
        ('--cpu-fixed-grid-source-qualification', 'fixed_grid_qualification', 'same-scope startup, effects and timing evidence'),
        ('--cpu-fixed-grid-forward', 'fixed_grid_forward', 'count-calibrated model and clean holdout review'),
    ):
        parser.add_argument(flag, dest=role, help=help_text)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(argv)
    evidence = {field.name: getattr(args, field.name) for field in fields(EntryEvidence)}
    evidence['launch_probes'] = tuple(evidence['launch_probes'])
    result=compile_build(args.domains,args.out,calibrations=args.calibration,
        no_op_profiles=None if args.no_op_profiles is None else json.loads(Path(args.no_op_profiles).read_bytes()),
        plan_only=args.plan_only,loop_trip_count=args.loop_trip_count,
        loop_inputs=None if args.loop_inputs is None else json.loads(Path(args.loop_inputs).read_bytes()),
        prepare_stateful_validation=args.prepare_stateful_validation,stateful_cpu_threads=args.stateful_cpu_threads,
        loop_cpu_threads=args.loop_cpu_threads,loop_cpu_model=args.loop_cpu_model,
        loop_calibrations=args.loop_calibration,
        loop_no_op_profiles=None if args.loop_no_op_profiles is None else json.loads(Path(args.loop_no_op_profiles).read_bytes()),
        loop_max_threads=args.loop_max_threads,
        stateful_interval_plans=None if args.stateful_interval_plans is None else json.loads(Path(args.stateful_interval_plans).read_bytes()),
        entry_evidence=EntryEvidence(**evidence), verify_entry=args.verify_entry,
        enforce_cpu_capacity=args.enforce_cpu_capacity)
    print(json.dumps(dict(regions=len(result['regions']),invariant_loops=len(result['invariant_loops']),
        composite_invariant_loops=len(result['composite_invariant_loops']),
        array_loops=len(result['array_loops']),
        statuses=[r['status'] for r in [*result['regions'],*result['invariant_loops'],*result['composite_invariant_loops'],
                                      *result['array_loops']]],
        whole_entry_status=result.get('whole_entry_status'),whole_entry_action=result.get('whole_entry_action'),
        result=str(Path(args.out)/'result.json'),seconds=result['seconds']),indent=2));return 0

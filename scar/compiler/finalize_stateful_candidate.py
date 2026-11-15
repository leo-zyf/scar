"""Export one source-proven, measured stateful candidate into a pipeline.

This does not select the best CPU mask, alter the complete-domain cost solver,
or qualify fresh-process entry performance. The existing native loader,
recipe validator, measured selector and exporter remain authoritative.
"""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256


def _path(value, filename):
    value=Path(value).resolve()
    return value/filename if value.is_dir() else value


def _save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    return artifact_reference(path,value['schema'])


def _stats(rows):
    if type(rows) is not list or not rows:raise ValueError('observed native execution statistics required')
    native=fallback=0
    for row in rows:
        a,b=row.get('native_calls'),row.get('fallback_calls')
        if type(a) is not int or type(b) is not int or a<0 or b<0:
            raise ValueError('exact nonnegative observed native/fallback counts required')
        native+=a;fallback+=b
    if native<=0 or fallback:raise ValueError('stateful validation must observe native execution with zero fallbacks')
    return native


def _validated_samples(measurement,report):
    if (measurement.get('native_execution_validated') is not True
            or measurement.get('original_source_modified') is not False
            or not measurement.get('correctness')
            or any(value is not True for value in measurement['correctness'].values())):
        raise ValueError('nonempty successful correctness report and unchanged source required')
    observed=measurement.get('artifact',{})
    if any(observed.get(key)!=report[key] for key in
            ('build_id','region_sha256','plan_sha256','contract_sha256')):
        raise ValueError('stateful validation names a different native artifact')
    if any(observed.get(key)!=value for key,value in report.items()):
        raise ValueError('validated native metadata differs from the fully admitted artifact')
    episodes=measurement.get('episodes')
    if type(episodes) is not list or not episodes:raise ValueError('nonempty observed validation episodes required')
    seen=set();measured=[];warm=0;native_calls=0
    for episode in episodes:
        index=episode.get('repetition')
        if type(index) is not int or index in seen:raise ValueError('unique exact repetition indices required')
        seen.add(index)
        if any(episode.get(key) is not True for key in ('bitwise_state','stdout_equal','file_bytes_equal')):
            raise ValueError('state, stdout or file-byte validation failed')
        native_calls+=_stats(episode.get('native_stats'))
        for key in ('baseline_seconds','native_seconds'):
            value=episode.get(key)
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                raise ValueError('positive finite observed episode timings required')
        if index<0:warm+=1
        else:measured.append(episode)
    measured.sort(key=lambda row:row['repetition'])
    repetitions=measurement.get('repetitions');warmup=measurement.get('warmup')
    if (type(repetitions) is not int or repetitions<3 or len(measured)!=repetitions
            or [row['repetition'] for row in measured]!=list(range(repetitions))
            or type(warmup) is not int or warmup<0 or warm!=warmup):
        raise ValueError('at least three complete measured pairs and declared warmup episodes required')
    baseline=[row['baseline_seconds'] for row in measured];native=[row['native_seconds'] for row in measured]
    for key,values in (('baseline_seconds',baseline),('optimized_seconds',native)):
        summary=measurement[key]
        if summary['samples']!=values:raise ValueError('summary samples differ from original episode clocks')
        expected=dict(median=statistics.median(values),min=min(values),max=max(values),stdev=statistics.stdev(values))
        if any(summary.get(name)!=value for name,value in expected.items()):
            raise ValueError('timing summary differs from original sample replay')
    if measurement.get('trajectory_stats') is not None:_stats(measurement['trajectory_stats'])
    return baseline,native,native_calls


def finalize_stateful_candidate(pipeline,candidate,validation,out,minimum_gain=.05):
    """Preserve the pipeline and add a measured, guarded stateful bundle row."""
    from scar.codegen.stateful import load
    from scar.compiler.export_stateful import export_stateful
    from scar.compiler.stateful_model import stateful_digest,stateful_from_dict
    from scar.optimization.selection import select_stateful_measured
    from scar.optimization.stateful_recipe import verify_stateful_recipe,unpack_stateful_recipe
    pipeline=_path(pipeline,'result.json');candidate=_path(candidate,'candidate.json')
    validation=_path(validation,'result.json');out=Path(out).resolve()
    if out.exists():raise ValueError('fresh finalized pipeline output required')
    if type(minimum_gain) not in (int,float) or not math.isfinite(minimum_gain) or not 0<=minimum_gain<1:
        raise ValueError('finite minimum gain in [0,1) required')
    pipeline_ref=artifact_reference(pipeline,'scar.program-compiler-pipeline');original=read_artifact(pipeline_ref)
    candidate_ref=artifact_reference(candidate,'scar.stateful-validation-candidate');document=read_artifact(candidate_ref)
    if original['schema_version']!=1 or document['schema_version']!=1:raise ValueError('supported pipeline and candidate versions required')
    if document['program']!=original['program']:raise ValueError('candidate belongs to another original program graph')
    read_artifact(original['program'])
    linked=[row for row in original['regions'] if row.get('validation_candidate')==candidate_ref]
    if len(linked)!=1:raise ValueError('candidate must match exactly one original pipeline validation_candidate reference')
    if (document.get('source_authority')!='ORIGINAL_F4_REDISCOVERY_EXACT_MATCH'
            or document.get('cost_domain_selection_unchanged') is not True
            or document.get('target_variants_ranked') is not False):
        raise ValueError('original F4 candidate admission and unchanged cost-domain selection required')
    region=stateful_from_dict(read_artifact(document['region']))
    if linked[0]['region_id']!=region.id:raise ValueError('candidate region differs from its original pipeline row')
    read_artifact(document['native_artifact']);read_artifact(document['recipe'])
    native=load(document['native_directory'])
    if artifact_reference(native.directory/'artifact.json',native.report['schema'])!=document['native_artifact']:
        raise ValueError('native directory differs from its candidate artifact reference')
    recipe=verify_stateful_recipe(document['recipe']['uri']);recipe_region,recipe_plan=unpack_stateful_recipe(recipe)
    if (stateful_digest(region)!=stateful_digest(native.region)
            or stateful_digest(recipe_region)!=stateful_digest(native.region)
            or semantic_sha256(recipe_plan)!=semantic_sha256(native.plan)
            or semantic_sha256(document['resource_binding'])!=semantic_sha256(native.plan['resource_binding'])):
        raise ValueError('candidate recipe, source region, storage plan or CPU binding differs from native replay')
    if out.is_relative_to(native.directory.resolve()):raise ValueError('finalization cannot modify the original native artifact')
    validation_document=json.loads(validation.read_bytes())
    schema=validation_document.get('schema')
    if type(schema) is not str or not schema:raise ValueError('versioned external validation report required')
    validation_ref=artifact_reference(validation,schema);measurement=read_artifact(validation_ref)
    baseline,optimized,native_calls=_validated_samples(measurement,native.report)
    selection=select_stateful_measured(native,baseline,optimized,validated=True,native_fallbacks=0,minimum_gain=minimum_gain)
    selection['evidence']=dict(path=str(validation),sha256=validation_ref['sha256'])
    out.mkdir(parents=True)
    selection_ref=_save(out/'selection.json',dict(schema='scar.stateful-measured-selection',schema_version=1,selection=selection,
        candidate=candidate_ref,validation=validation_ref,
        fresh_process_scope_measured=False,full_entry_performance_qualified=False,target_variants_ranked=False))
    result=deepcopy(original);stateful=list(result.get('stateful_regions',()))
    if any(row['id']==region.id for row in stateful):raise ValueError('pipeline already contains a finalized row for this original region')
    row=dict(id=region.id,status='MEASURED_STATEFUL_BUNDLE_READY' if selection['selected'] else 'MEASURED_STATEFUL_CANDIDATE_NOT_PROFITABLE',
        proof=candidate_ref,schedule=document['recipe'],selection=selection_ref,validation=validation_ref,
        input_shapes=native.plan['input_shapes'],trip_count=None,memory_budget_bytes=native.plan['memory_budget_bytes'],
        selected_kind='ORDERED_STATEFUL_REGION',native_directory=str(native.directory),native_artifact=document['native_artifact'],
        source_authority=document['source_authority'],resource_binding=document['resource_binding'],
        cost_domain_selection_unchanged=True,full_entry_performance_qualified=False,
        fresh_process_scope_measured=False,measured_region_selected=selection['selected'],
        selected_for_use=False,whole_program_cpp=False,physical_optimality_proven=False)
    if selection['selected']:
        exported=export_stateful(native,out/'bundle',selection_path=out/'selection.json')
        row.update(bundle=exported['bundle'],bundle_export=artifact_reference(out/'bundle/export-report.json',exported['schema']))
    stateful.append(row);result['stateful_regions']=stateful
    result['stateful_finalization']=dict(parent_pipeline=pipeline_ref,candidate=candidate_ref,validation=validation_ref,
        selection=selection_ref,validated_native_calls=native_calls,original_regions_and_invariant_loops_preserved=True,
        full_entry_performance_qualified=False,physical_optimality_proven=False)
    # Export replays selection and all original native admission again. Recheck
    # external inputs before publishing this additive pipeline packet.
    for reference in (pipeline_ref,candidate_ref,validation_ref):read_artifact(reference)
    _save(out/'result.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline',required=True);parser.add_argument('--candidate',required=True)
    parser.add_argument('--validation',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--minimum-gain',type=float,default=.05);args=parser.parse_args(argv)
    result=finalize_stateful_candidate(args.pipeline,args.candidate,args.validation,args.out,args.minimum_gain)
    row=result['stateful_regions'][-1]
    print(json.dumps(dict(result=str(Path(args.out).resolve()/'result.json'),status=row['status'],
        bundle=row.get('bundle'),full_entry_performance_qualified=False,physical_optimality_proven=False),sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())

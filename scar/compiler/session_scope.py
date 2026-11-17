"""Discover source callers, discharge effect guards and export scoped entry delivery."""
import argparse
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA='scar.workspace-session-scope-preparation'


def _write(path,value):
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    return artifact_reference(path,value['schema'])


def _pins():
    root=Path(__file__).resolve().parents[1]
    return {n:artifact_reference(root/n,'scar.python-source') for n in
        ('compiler/session_scope.py','compiler/inspect_workspace_session.py',
         'analysis/session_caller_effects.py','optimization/session_caller.py','optimization/session_lifetime.py')}


def _derive(compiled_reference,lifetime_reference):
    from scar.compiler.inspect_workspace_session import inspect_workspace_session
    from scar.analysis.build import BuildSession
    from scar.analysis.source_body_interfaces import BodyInterfaceResolver
    from scar.analysis.source_call_view import SourceFunctionCallProvider
    from scar.analysis.source_reference_slice import ReferenceSlicer
    from scar.analysis.session_caller_effects import _source
    from scar.optimization.session_caller import derive_guarded_caller,_identity
    from scar.optimization.session_lifetime import verify_session_lifetime
    from scar.ir.versioned_schedule import schedule_from_dict
    from scar.ir.temporal_resources import projection_from_dict
    from scar.ir.program import normal_links as F
    from scar.ir.record_codec import encode
    check,context=inspect_workspace_session(compiled_reference['uri'],with_context=True)
    if check['compiled'] != compiled_reference:raise ValueError('session compiled source changed')
    contract=context['compact'];step=contract['step_contract'];program=context['preparation']['program']
    lifetime=read_artifact(lifetime_reference);parents=lifetime['parents']
    if parents['session_native'] != check['native']:raise ValueError('lease lifetime belongs to another native session')
    verify_session_lifetime(context['plan'],schedule_from_dict(read_artifact(parents['kernel_schedule'])),
        projection_from_dict(read_artifact(parents['kernel_resources'])),lifetime,
        session_plan_ref=context['preparation']['session_plan'],kernel_schedule_ref=parents['kernel_schedule'],
        kernel_resources_ref=parents['kernel_resources'],native_reference=check['native'],native_metadata=context['native'],
        parent_native_metadata=context['original']['native_metadata'],source_refs=lifetime['source_refs'])
    profiles=[];ledgers=[];frontiers=[]
    with BuildSession(Path(program['uri']).parent) as built:
        if read_artifact(program) != built.report:raise ValueError('caller source build changed')
        resolver=BodyInterfaceResolver(built.view)
        for unit in sorted(built.view.catalog.units_for_source(step['source']['path']) or ()):
            original=built.view.unit(unit);facts=original.facts
            kernels=[i for i,f in enumerate(facts.functions)
                if _identity(encode(_source(built.view,unit,f)[0])) == _identity(step['source'])]
            if len(kernels)!=1:continue
            slicer=ReferenceSlicer(SimpleNamespace(fact_store=built.view.catalog.fact_store))
            kernel_fact=slicer.fact(facts,'functions',kernels[0])
            selected_calls=set()
            # Targets only select callers; the per-site direct lookup proof
            # and live source guards, not observation frequency, admit reuse.
            fact_to_ordinal={slicer.fact(facts,'calls',i):i for i in range(len(facts.calls))}
            for provider in built._f4.providers.values():
                if not isinstance(provider,SourceFunctionCallProvider) or unit not in provider.available_units:continue
                graph=provider.store.read_unit(unit)
                for interface in graph.interfaces:
                    if interface.definition == kernel_fact and interface.call in fact_to_ordinal:
                        selected_calls.add(fact_to_ordinal[interface.call])
            for ordinal,function in enumerate(facts.functions):
                if function.body_scope is None:continue
                indices={i for g in facts.scope_groups if g.owner==function.body_scope.owner
                    for span in g.tables if span.table is F.FactTable.CALLS for i in range(span.start,span.stop)}
                if not indices & selected_calls:continue
                profile,ledger=derive_guarded_caller(built.view,unit=unit,function_ordinal=ordinal,
                    step_contract=step,copy_effects=contract['copy_effects'],resolver=resolver)
                if profile['status']=='GUARDED_EFFECT_PROFILE_CLOSED':profiles.append(profile);ledgers.append(ledger)
                else:frontiers.append(profile)
    boundaries={}
    for profile in profiles:
        children={call['boundary']['id']:call['boundary']for call in profile['calls']}
        source=profile['source'];row=dict(id=profile['id'],role='CALLER',source=source,
            owner_path=profile['owner_path'],attribute=source['qualname'].split('.')[-1],descriptor='METHOD',
            parameters=profile['parameters'],guards=profile['guards'],allowed_call_roles=sorted(children),
            lookup_routes=[dict(path=[profile['parameters'][0],c['boundary']['attribute']],
                boundary_id=c['boundary']['id'])for c in profile['calls']],
            proof=dict(structural_operations_complete=True,effect_profile_closed=True,unknown_calls_barriered=True),
            retain_arguments_fields_and_descriptors_until_outer_exit=True)
        for key,value in {**children,row['id']:row}.items():
            if key in boundaries and boundaries[key]!=value:raise ValueError('conflicting source boundary installation')
            boundaries[key]=value
    spec=dict(schema='scar.workspace-session-scope',schema_version=1,
        boundaries=[boundaries[k]for k in sorted(boundaries)],caller_profile_ids=[p['id']for p in profiles],
        cross_activation_residency_authorized=False,outer_entry_invalidates_and_exit_destroys=True,
        unknown_effect_taints_all_active_callers_until_outer_exit=True,
        default_backend='NO_OP',trial_only=True,deployment_authorized=False)
    return spec,profiles,ledgers,frontiers,check,lifetime


def _report(compiled,lifetime,spec_ref,profiles_ref,ledger_refs,frontiers,profiles):
    return dict(schema=SCHEMA,schema_version=1,compiled_session=compiled,session_lifetime=lifetime,
        scope=spec_ref,profiles=profiles_ref,ledgers=ledger_refs,frontiers=frontiers,implementations=_pins(),
        summary=dict(closed_callers=len(profiles),operations=sum(len(p['obligations'])for p in profiles)),
        cross_activation_residency_authorized=False,target_program_executed=False,
        original_caller_control_retained=True,default_backend='NO_OP',deployment_authorized=False)


def prepare(compiled,lifetime,out):
    compiled=artifact_reference(Path(compiled).resolve(),'scar.compiled-workspace-session')
    lifetime=artifact_reference(Path(lifetime).resolve(),'scar.workspace-session-lifetime')
    output=Path(out).resolve()
    if output.exists():raise ValueError('fresh scope preparation output required')
    spec,profiles,ledgers,frontiers,_,_=_derive(compiled,lifetime)
    output.mkdir(parents=True)
    lrefs=[_write(output/f'caller-{i}-effects.json',ledger)for i,ledger in enumerate(ledgers)]
    pdoc=dict(schema='scar.guarded-session-callers',schema_version=1,profiles=profiles)
    report=_report(compiled,lifetime,_write(output/'scope.json',spec),_write(output/'profiles.json',pdoc),
        lrefs,frontiers,profiles)
    _write(output/'result.json',report);return report


def replay(path):
    report=read_artifact(artifact_reference(Path(path).resolve(),SCHEMA))
    spec,profiles,ledgers,frontiers,check,lifetime=_derive(report['compiled_session'],report['session_lifetime'])
    expected=_report(report['compiled_session'],report['session_lifetime'],report['scope'],report['profiles'],
        report['ledgers'],frontiers,profiles)
    if (report!=expected or read_artifact(report['scope'])!=spec
            or read_artifact(report['profiles'])!=dict(schema='scar.guarded-session-callers',schema_version=1,profiles=profiles)
            or len(report['ledgers'])!=len(ledgers)
            or any(read_artifact(r)!=d for r,d in zip(report['ledgers'],ledgers))):
        raise ValueError('scope/caller effects changed from source replay')
    return report,spec,check,lifetime


def export(preparation,out):
    from scar.runtime.session_scope_bundle import ADMISSION,implementation_sha256,inspect_scope_bundle
    report,spec,check,lifetime=replay(preparation)
    if not report['summary']['closed_callers']:raise ValueError('no source-qualified caller to install')
    output=Path(out).resolve()
    if output.exists():raise ValueError('fresh scoped runtime export required')
    original=read_artifact(report['compiled_session'])
    shutil.copytree(original['bundle'],output)
    scope_ref=_write(output/'scope.json',spec)
    admission=dict(schema=ADMISSION,schema_version=1,scope_sha256=scope_ref['sha256'],
        runtime_implementation_sha256=implementation_sha256(),
        session_admission_sha256=original['expected_admission_sha256'],
        preparation=artifact_reference(Path(preparation).resolve(),SCHEMA),
        lifetime_projection_sha256=report['session_lifetime']['sha256'])
    pin=_write(output/'scope-admission.json',admission)
    inspect_scope_bundle(output,expected_admission_sha256=pin['sha256'])
    result=dict(schema='scar.workspace-session-scope-export',schema_version=1,bundle=str(output),
        expected_admission_sha256=pin['sha256'],preparation=admission['preparation'],
        boundaries=len(spec['boundaries']),cross_activation_residency_authorized=False,
        native_recompiled=False,default_backend='NO_OP',deployment_authorized=False)
    _write(output/'scope-export.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--session');group.add_argument('--export');group.add_argument('--check')
    parser.add_argument('--lifetime');parser.add_argument('--out');args=parser.parse_args(argv)
    if args.check:result=replay(args.check)[0]
    elif args.export:
        if not args.out:parser.error('export requires --out')
        result=export(args.export,args.out)
    else:
        if not args.lifetime or not args.out:parser.error('session preparation requires --lifetime and --out')
        result=prepare(args.session,args.lifetime,args.out)
    print(json.dumps({k:result[k]for k in ('summary','bundle','expected_admission_sha256','boundaries')if k in result},sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())

"""Publish shared outer-loop models and isolated native-leaf entry delivery."""
import argparse
import json
from pathlib import Path
import shutil

from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA='scar.session-outer-preparation'


def _write(path,value):
    with path.open('x')as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return artifact_reference(path,value['schema'])


def _pins():
    root=Path(__file__).resolve().parents[1]
    return {n:artifact_reference(root/n,'scar.python-source')for n in
        ('compiler/session_outer.py','optimization/session_outer.py','analysis/session_scalar_flow.py',
         'optimization/session_activation.py')}


def _derive(reference):
    from scar.compiler.session_scope import replay
    from scar.analysis.build import BuildSession
    from scar.analysis.source_body_interfaces import BodyInterfaceResolver
    from scar.analysis.session_caller_effects import _source
    from scar.compiler.stateful_frontend import _reachable
    from scar.optimization.session_outer import derive_session_outer
    from scar.ir.program import normal_links as F
    from scar.ir.record_codec import encode
    report,spec,check,lifetime=replay(reference['uri'])
    if report!=read_artifact(reference):raise ValueError('nested source caller preparation changed')
    nested=read_artifact(report['profiles'])['profiles']
    session=read_artifact(read_artifact(report['compiled_session'])['preparation'])
    profiles=[];scalars=[];frontiers=[]
    with BuildSession(Path(session['program']['uri']).parent)as built:
        if read_artifact(session['program'])!=built.report:raise ValueError('outer source build changed')
        resolver=BodyInterfaceResolver(built.view)
        by_unit={}
        for p in nested:by_unit.setdefault(p['unit'],[]).append(p)
        for unit,children in by_unit.items():
            original=built.view.unit(unit);facts=original.facts
            names={p['source']['qualname'].split('.')[-1]for p in children}
            attrs={alias:a for a in facts.attributes if a.action is F.AttributeAction.GET and a.name in names
                for alias in _reachable(facts,a.value)}
            for ordinal,fn in enumerate(facts.functions):
                if fn.body_scope is None:continue
                indices={i for g in facts.scope_groups if g.owner==fn.body_scope.owner
                    for span in g.tables if span.table is F.FactTable.CALLS for i in range(span.start,span.stop)}
                if not any(c.callee in attrs and attrs[c.callee].name in names for c in (facts.calls[i]for i in indices)):
                    continue
                source=encode(_source(built.view,unit,fn)[0])
                if any(source['code_key']==p['source']['code_key']for p in children):continue
                profile,scalar=derive_session_outer(built.view,unit=unit,function_ordinal=ordinal,
                    nested_profiles=children,resolver=resolver)
                if profile['status']=='FRONTIER':frontiers.append(profile)
                else:profiles.append(profile);scalars.append(scalar)
    # Unknown host lookups remain untouched. Because outer effect closure is
    # not proven, these activations never authorize inter-call retention.
    combined=dict(spec);boundaries=list(spec['boundaries'])
    for profile in profiles:
        routes=[dict(path=[profile['parameters'][0],c['boundary']['attribute']],
            boundary_id=c['boundary']['id'])for c in profile['calls']if c['role']=='NESTED_CALLER']
        boundaries.append(dict(id=profile['id'],role='CALLER',source=profile['source'],
            owner_path=profile['owner_path'],attribute=profile['source']['qualname'].split('.')[-1],descriptor='METHOD',
            parameters=profile['parameters'],guards=[],lookup_routes=routes,
            allowed_call_roles=profile['nested_caller_ids'],
            proof=dict(structural_operations_complete=True,effect_profile_closed=False,unknown_calls_barriered=False)))
    combined.update(boundaries=boundaries,unqualified_kernel_policy='FRESH_ISOLATED_NATIVE',
        outer_profile_ids=[p['id']for p in profiles],
        unknown_host_lookup_policy='ORIGINAL_UNCHANGED_NO_LIVE_LEASE',
        opaque_host_quiescence_assumed=False)
    return report,combined,profiles,scalars,frontiers,check,lifetime


def _report(reference,scope_ref,profile_refs,scalar_refs,activation_refs,frontiers,profiles):
    return dict(schema=SCHEMA,schema_version=1,nested_preparation=reference,scope=scope_ref,
        outer_profiles=profile_refs,scalar_flows=scalar_refs,activations=activation_refs,frontiers=frontiers,
        implementations=_pins(),summary=dict(outer_callers=len(profiles),
            source_operations=sum(p['operations']for p in profiles),shared_loops=sum(len(p['shared_loops'])for p in profiles)),
        original_host_calls_and_control_retained=True,effect_profile_closed=False,
        retention_authorized=False,opaque_host_quiescence_assumed=False,
        target_program_executed=False,default_backend='NO_OP',deployment_authorized=False)


def prepare(nested_preparation,out):
    from scar.optimization.session_activation import derive_session_activation
    ref=artifact_reference(Path(nested_preparation).resolve(),'scar.workspace-session-scope-preparation')
    output=Path(out).resolve()
    if output.exists():raise ValueError('fresh outer preparation required')
    parent,spec,profiles,scalars,frontiers,_,lifetime=_derive(ref)
    output.mkdir(parents=True)
    p_refs=[];s_refs=[];a_refs=[]
    for i,(profile,scalar)in enumerate(zip(profiles,scalars)):
        p_refs.append(_write(output/f'outer-{i}.json',profile))
        s_refs.append(_write(output/f'scalar-{i}.json',scalar))
        activation=derive_session_activation(profile,lifetime,outer_profile_ref=p_refs[-1],
            lifetime_ref=parent['session_lifetime'])
        a_refs.append(_write(output/f'activation-{i}.json',activation))
    result=_report(ref,_write(output/'scope.json',spec),p_refs,s_refs,a_refs,frontiers,profiles)
    _write(output/'result.json',result);return result


def replay(path):
    from scar.optimization.session_activation import verify_session_activation
    report=read_artifact(artifact_reference(Path(path).resolve(),SCHEMA))
    parent,spec,profiles,scalars,frontiers,check,lifetime=_derive(report['nested_preparation'])
    if (report!=_report(report['nested_preparation'],report['scope'],report['outer_profiles'],report['scalar_flows'],
            report['activations'],frontiers,profiles) or read_artifact(report['scope'])!=spec
            or len(profiles)!=len(report['outer_profiles'])or len(scalars)!=len(report['scalar_flows'])
            or len(profiles)!=len(report['activations'])):raise ValueError('outer source preparation differs')
    for profile,scalar,p,s,a in zip(profiles,scalars,report['outer_profiles'],report['scalar_flows'],report['activations']):
        if read_artifact(p)!=profile or read_artifact(s)!=scalar:raise ValueError('outer scalar/reference/control graph changed')
        verify_session_activation(profile,lifetime,read_artifact(a),outer_profile_ref=p,lifetime_ref=parent['session_lifetime'])
    return report,spec,parent,check


def export(preparation,out):
    from scar.runtime.session_scope_bundle import ADMISSION,implementation_sha256,inspect_scope_bundle
    report,spec,parent,_=replay(preparation)
    if not report['summary']['outer_callers']:raise ValueError('no shared outer source loop found')
    output=Path(out).resolve()
    if output.exists():raise ValueError('fresh outer scope bundle required')
    native=read_artifact(parent['compiled_session']);shutil.copytree(native['bundle'],output)
    scope=_write(output/'scope.json',spec)
    admission=dict(schema=ADMISSION,schema_version=1,scope_sha256=scope['sha256'],
        runtime_implementation_sha256=implementation_sha256(),session_admission_sha256=native['expected_admission_sha256'],
        preparation=artifact_reference(Path(preparation).resolve(),SCHEMA),retention_authorized=False)
    pin=_write(output/'scope-admission.json',admission)
    inspect_scope_bundle(output,expected_admission_sha256=pin['sha256'])
    result=dict(schema='scar.session-outer-export',schema_version=1,bundle=str(output),
        preparation=admission['preparation'],expected_admission_sha256=pin['sha256'],
        original_host_calls_retained=True,native_recompiled=False,retention_authorized=False,
        default_backend='NO_OP',deployment_authorized=False)
    _write(output/'scope-export.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--nested');source.add_argument('--export');source.add_argument('--check');parser.add_argument('--out')
    args=parser.parse_args(argv)
    if args.check:result=replay(args.check)[0]
    elif not args.out:parser.error('preparation/export requires --out')
    elif args.export:result=export(args.export,args.out)
    else:result=prepare(args.nested,args.out)
    print(json.dumps({k:result[k]for k in ('summary','bundle','expected_admission_sha256')if k in result},sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())

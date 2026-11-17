"""Source-admitted copy effects and a dynamic, explicitly invalidated C++ lease."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256

SCHEMA='scar.workspace-session-preparation'
NATIVE='scar.native-workspace-session'


def _write(path,value):
    with path.open('x')as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return artifact_reference(path,value['schema'])


def _pins():
    root=Path(__file__).resolve().parents[1]
    return {name:artifact_reference(root/name,'scar.python-source')for name in (
        'compiler/workspace_session.py','analysis/array_copy_effects.py','optimization/residency_session.py',
        'ir/workspace_session.py','codegen/workspace_session.py')}


def _derive(residency_reference,program_reference):
    from scar.compiler.compile_workspace_shadow import replay_workspace_shadow
    from scar.optimization.residency_session import derive_workspace_session,copy_effect_preserves_session
    from scar.analysis.array_copy_effects import derive_array_copy_effects
    from scar.analysis.build import BuildSession
    from scar.analysis.source_body_interfaces import BodyInterfaceResolver
    report,residency,context=replay_workspace_shadow(residency_reference['uri'])
    if read_artifact(residency_reference)!=report:raise ValueError('parent residency changed')
    session_plan=derive_workspace_session(residency,context['step_plan'],context['native_metadata'],
        source_refs=(dict(artifact=residency_reference,locator='#/residency_plan'),))
    shapes={tuple(s['shape'])for s in residency['snapshots']}
    effects=[];frontiers=[]
    # Select by the parent's source path and original formal ordinals. Names
    # are presentation metadata, never optimization rules.
    parent=Path(context['native_manifest']['uri']).parent
    region=read_artifact(artifact_reference(parent/context['native_metadata']['files']['region'],'scar.stateful-array-region'))
    with BuildSession(Path(program_reference['uri']).parent)as built:
        if read_artifact(program_reference)!=built.report:raise ValueError('source program build changed')
        resolver=BodyInterfaceResolver(built.view)
        for unit in sorted(built.view.catalog.units_for_source(region['source']['path'])or()):
            original=built.view.unit(unit)
            for ordinal,function in enumerate(original.facts.functions):
                if len(function.parameter_names)!=1:continue
                for shape in sorted(shapes):
                    effect=derive_array_copy_effects(built.view,unit=unit,function_ordinal=ordinal,
                        input_shapes={0:shape},resolver=resolver,source_refs=(dict(artifact=program_reference,locator='#'),))
                    if effect['status']=='CLOSED_COPY_EFFECTS_CONDITIONAL' and effect['copies']:
                        compatibility=[copy_effect_preserves_session(session_plan,effect,formal_index=0,input_id=r['input_id'])
                            for r in session_plan['geometry']['snapshots'] if r['shape']==list(shape)
                            and r['mode']!='KEEP_FULL_SNAPSHOT']
                        effects.append(dict(summary=effect,compatibility=compatibility))
                    elif effect['frontiers']:
                        frontiers.append(dict(unit=unit,function_ordinal=ordinal,frontiers=effect['frontiers']))
    read_artifact(program_reference);read_artifact(residency_reference)
    return session_plan,effects,frontiers,context,residency


def _report(residency,program,plan_ref,effects_ref,frontiers,plan,effects):
    return dict(schema=SCHEMA,schema_version=1,residency_preparation=residency,program=program,
        session_plan=plan_ref,copy_effects=effects_ref,frontiers=frontiers,implementations=_pins(),
        summary=dict(closed_copy_summaries=len(effects),protected_roots=len(plan['geometry']['snapshots'])),
        source_caller_effect_coverage_proven=False,caller_replacement_authorized=False,
        target_program_executed=False,native_compiled=False,default_backend='NO_OP',deployment_authorized=False)


def prepare_session(residency,program,out):
    ref=artifact_reference(Path(residency).resolve(),'scar.workspace-shadow-residency-preparation')
    program=Path(program).resolve();program=program/'manifest.json'if program.is_dir()else program
    p=artifact_reference(program,'scar.program-build');directory=Path(out).resolve()
    if directory.exists():raise ValueError('fresh session preparation required')
    plan,effects,frontiers,_,_=_derive(ref,p);directory.mkdir(parents=True)
    effect_doc=dict(schema='scar.workspace-session-copy-effects',schema_version=1,effects=effects)
    report=_report(ref,p,_write(directory/'session.json',plan),_write(directory/'effects.json',effect_doc),
        frontiers,plan,effects)
    _write(directory/'result.json',report);return report


def replay_session(path):
    reference=artifact_reference(Path(path).resolve(),SCHEMA);report=read_artifact(reference)
    plan,effects,frontiers,context,residency=_derive(report['residency_preparation'],report['program'])
    expected=_report(report['residency_preparation'],report['program'],report['session_plan'],report['copy_effects'],
        frontiers,plan,effects)
    if (report!=expected or read_artifact(report['session_plan'])!=plan
            or read_artifact(report['copy_effects'])!=dict(schema='scar.workspace-session-copy-effects',schema_version=1,effects=effects)):
        raise ValueError('session/source copy effects differ from full replay')
    read_artifact(reference);return reference,report,plan,effects,context,residency


def compile_session(preparation,out):
    from scar.codegen.workspace_session import render_workspace_session
    from scar.compiler.stateful_model import stateful_from_dict
    from scar.compiler.export_stateful import project_stateful_contract
    from scar.runtime.native_workspace_session import SCHEMA as RUNTIME,ADMISSION,implementation_sha256,inspect_bundle
    ref,report,plan,effects,context,residency=replay_session(preparation)
    source=read_artifact(plan['native_blueprint']['cpp_source'],json_document=False)
    output=Path(out).resolve()
    if output.exists():raise ValueError('fresh native session output required')
    cpp,mapping=render_workspace_session(source,context['native_metadata'],residency,plan)
    native=output/'native';native.mkdir(parents=True);(native/'session.cpp').write_bytes(cpp)
    parent=Path(context['native_manifest']['uri']).parent
    for name in ('stateful_abi_v1.h','scar_abi_v1.h'):
        shutil.copyfile(parent/context['native_metadata']['files'][name],native/name)
    compiler=context['native_metadata']['compiler'];library=native/('session_'+mapping['execution_plan_sha256'][:20]+'.so')
    proc=subprocess.run([*compiler['argv'],*compiler['options'],str(native/'session.cpp'),'-o',str(library),'-lm'],capture_output=True,text=True)
    (native/'compiler-stderr.txt').write_text(proc.stderr)
    if proc.returncode:raise RuntimeError('session C++ compilation failed: '+proc.stderr)
    metadata=dict(schema=NATIVE,schema_version=1,preparation=ref,session_plan=report['session_plan'],
        source_mapping=mapping,execution_plan_sha256=mapping['execution_plan_sha256'],compiler=compiler,
        source=artifact_reference(native/'session.cpp','scar.cpp-source'),library=artifact_reference(library,'scar.native-library'))
    native_ref=_write(native/'artifact.json',metadata)
    base_ref=context['native_metadata']['base_native_reference'];base_meta=read_artifact(base_ref);base_dir=Path(base_ref['uri']).parent
    base_region=stateful_from_dict(read_artifact(artifact_reference(base_dir/base_meta['files']['region'],'scar.stateful-array-region')))
    step=project_stateful_contract(SimpleNamespace(directory=base_dir,region=base_region,
        plan=context['step_plan'],report=base_meta),selection=None,require_selection=False)
    bundle=output/'bundle';bundle.mkdir();base_lib=base_dir/base_meta['files']['library']
    def lib(path):return dict(library=path.name,sha256=sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
    compact_effects=[]
    for row in effects:
        effect=row['summary']
        compact_effects.append(dict(source=effect['source'],shape=effect['roots'][0]['shape'],summary_id=effect['id'],
            writes=[a['rectangle']for a in effect['accesses']if a['kind']=='WRITE_COPY_DESTINATION']))
    contract=dict(schema=RUNTIME,schema_version=1,step_contract=step,protocol=plan['protocol'],geometry=plan['geometry'],
        copy_effects=compact_effects,caller_replacement_authorized=False,
        native=dict(base=lib(base_lib),session=lib(library),symbols=plan['symbols'],execution_plan_sha256=mapping['execution_plan_sha256']),
        provider_boundary_Q=plan['provider_boundary_Q'],default_backend='NO_OP',trial_only=True)
    runtime_ref=_write(bundle/'runtime.json',contract)
    admission=dict(schema=ADMISSION,schema_version=1,runtime_sha256=runtime_ref['sha256'],
        runtime_implementation_sha256=implementation_sha256(),native=contract['native'],preparation=ref,
        native_manifest=native_ref,session_plan=report['session_plan'],caller_replacement_authorized=False)
    pin=_write(bundle/'admission.json',admission)
    shutil.copyfile(base_lib,bundle/base_lib.name);shutil.copyfile(library,bundle/library.name)
    inspect_bundle(bundle,expected_admission_sha256=pin['sha256'])
    result=dict(schema='scar.compiled-workspace-session',schema_version=1,preparation=ref,native=native_ref,
        bundle=str(bundle),expected_admission_sha256=pin['sha256'],source_caller_effect_coverage_proven=False,
        caller_replacement_authorized=False,default_backend='NO_OP',deployment_authorized=False)
    _write(output/'result.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--residency');group.add_argument('--preparation');group.add_argument('--check')
    parser.add_argument('--program');parser.add_argument('--out');args=parser.parse_args(argv)
    if args.check:
        if args.out:parser.error('check only consumes a saved preparation')
        result=replay_session(args.check)[1]
    elif args.preparation:
        if not args.out:parser.error('compilation requires --out')
        result=compile_session(args.preparation,args.out)
    else:
        if not args.out or not args.program:parser.error('source/effect preparation requires --program and --out')
        result=prepare_session(args.residency,args.program,args.out)
    print(json.dumps({k:result[k]for k in ('summary','bundle','expected_admission_sha256',
        'source_caller_effect_coverage_proven','caller_replacement_authorized')if k in result},sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())

"""Derive hardware planning domains directly from an admitted program graph.

This bridge consumes original F4 operations, world ports and the build's full
hardware snapshot. Unsupported functions stay explicit; no AST fallback or
target execution supplies a replacement semantic model.
"""
import argparse
import json
from pathlib import Path
import time


def plan_build(program, out, *, source=None, shape=None, input_shapes=None,
               memory_budget_bytes, affinity=None, max_threads=None):
    from scar.analysis.build import BuildSession,_check
    from scar.analysis.build_f4 import _validate_catalog_parents
    from scar.compiler.stateful_frontend import discover_stateful_regions
    from scar.compiler.stateful_model import write_stateful
    from scar.ir.execution_recipe import artifact_reference,read_artifact
    from scar.optimization.stateful_cpu_choices import build_stateful_cpu_domain
    program=Path(program).absolute();out=Path(out).absolute()
    if shape is not None and input_shapes is not None:
        raise ValueError('choose one common shape or per-region named input shapes')
    if shape is None and input_shapes is None:
        raise ValueError('declared or previously observed input shapes required')
    if out.exists():raise ValueError('fresh program planning output required')
    started=time.monotonic()
    parent=artifact_reference(program/'manifest.json','scar.program-build')
    manifest=read_artifact(parent)
    if manifest.get('backend')!='F4':raise ValueError('original F4 program build required')
    source=Path(source or manifest['entry']).absolute()
    with BuildSession(program) as session:
        target=manifest.get('target')
        if target is None:raise ValueError('program build has no target hardware snapshot')
        _check(target)
        hardware=Path(target['path']).parent
        # Reuse this session's catalog and original shared function bodies.
        regions,rejected=discover_stateful_regions(session.view,source=source)
        out.mkdir(parents=True)
        rows=[]
        for index,region in enumerate(regions):
            folder=out/('region-'+str(index));folder.mkdir()
            path=folder/'region.json';write_stateful(region,path)
            shapes=({value.id:list(shape) for value in region.inputs if value.kind=='array'}
                if shape is not None else input_shapes.get(region.id))
            row=dict(region_id=region.id,source_code_key=region.source.code_key,
                region=str(path),source_unit_scope='DECLARED_FUNCTION_PROFILE_NOT_RUNTIME_REACHABILITY',
                selected_for_use=False,lowering_authorized=False)
            if shapes is None:
                row.update(status='INPUT_SHAPES_REQUIRED')
            else:
                domain=build_stateful_cpu_domain(path,hardware,input_shapes=shapes,
                    memory_budget_bytes=memory_budget_bytes,affinity=affinity,
                    max_threads=max_threads,out=folder/'domain')
                row.update(status='HARDWARE_DOMAIN_BUILT',domain=str(folder/'domain/domain.json'),
                    domain_id=domain['domain_id'],candidate_count=len(domain['candidates']),
                    retained_hardware_resources=len(domain['all_resources']))
            rows.append(row)
        _validate_catalog_parents(session.view.catalog)
        _check(target)
        if artifact_reference(program/'manifest.json',parent['schema'])!=parent:
            raise ValueError('program build changed during graph planning')
    report=dict(schema='scar.program-optimization-domains',schema_version=1,
        program=parent,source=str(source),hardware=target,regions=rows,rejections=rejected,
        semantics='ORIGINAL_F4_FUNCTION_WORLD_AND_SOURCE_PORTS',
        catalog_reader_reused_for_region_discovery=True,
        source_ast_fallback_used=False,target_program_executed=False,
        target_variants_compiled=False,whole_program_compiled=False,
        first_stage_complete=False,seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',required=True);parser.add_argument('--source')
    shapes=parser.add_mutually_exclusive_group(required=True)
    shapes.add_argument('--shape',type=int,nargs=2)
    shapes.add_argument('--input-shapes',help='JSON mapping region IDs to array-ID shape maps')
    parser.add_argument('--memory-budget-bytes',type=int,required=True)
    parser.add_argument('--max-threads',type=int);parser.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    result=plan_build(args.build,args.out,source=args.source,shape=args.shape,
        input_shapes=None if args.input_shapes is None else json.loads(Path(args.input_shapes).read_bytes()),
        memory_budget_bytes=args.memory_budget_bytes,max_threads=args.max_threads)
    print(json.dumps(dict(regions=len(result['regions']),rejections=len(result['rejections']),
        result=str(Path(args.out)/'result.json'),seconds=result['seconds']),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())

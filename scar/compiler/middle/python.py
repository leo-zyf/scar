"""Derive hardware planning domains directly from an admitted program graph.

This bridge consumes original F4 operations, world ports and the build's full
hardware snapshot. Unsupported functions stay explicit; no AST fallback or
target execution supplies a replacement semantic model.
"""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

from scar.analysis.source_decoded_bodies import reuse_decoded_source_bodies


@dataclass(frozen=True, slots=True)
class ScheduledLoop:
    loop_id: str
    folder: Path
    proof: dict
    source_motion: dict
    schedule: dict
    composite: bool
    parallel: bool


def schedule_loops(document, out, catalog, *, trip_count=None, input_specs=None, cpu_threads=None,
                   cpu_model=False, calibrations=(), original_costs=None, max_threads=None):
    """Project proven loop movement and publish schedules independently of codegen."""
    from scar.compiler.loop_admission import read_invariant_loop_motion, validate_invariant_loop_links
    from scar.compiler.pipeline_loop_frontend import (
        read_composite_invariant_loop_motion, validate_composite_invariant_loop_links,
    )
    from scar.ir.execution_recipe import read_artifact
    from .plan import plan_loop_region, publish_plan

    input_specs = input_specs or {}
    original_costs = original_costs or {}
    profiles = (
        ('invariant_loops', 'loops', read_invariant_loop_motion, validate_invariant_loop_links),
        ('composite_invariant_loops', 'composite-loops', read_composite_invariant_loop_motion,
         validate_composite_invariant_loop_links),
    )
    known = {spec['id'] for name, *_ in profiles for spec in document.get(name, ())}
    if (input_specs.keys() | original_costs.keys()) - known:
        raise ValueError('loop inputs and original costs must name retained loop IDs')
    direct, composite, work = [], [], []
    if not known:
        return direct, composite, work
    if not cpu_model and (calibrations or original_costs or max_threads is not None):
        raise ValueError('loop calibration and CPU model limits require --loop-cpu-model')
    if cpu_model and cpu_threads is not None:
        raise ValueError('choose automatic CPU selection or an explicit team preview')
    hardware = document['hardware']
    for (name, subdirectory, read_motion, check_links), rows in zip(profiles, (direct, composite)):
        for index, spec in enumerate(document.get(name, ())):
            folder = out / subdirectory / f'loop-{index}'
            folder.mkdir(parents=True)
            reference = spec['proof']
            read_artifact(reference)
            proof, motion = read_motion(reference['uri'], catalog=catalog)
            if spec['id'] != proof.id or spec['hardware'] != hardware:
                raise ValueError('loop proof or hardware differs from the original program domain')
            check_links(proof, spec, document['loop_dependencies'])
            composed = rows is composite
            family = 'COMPOSITE_LOOP' if composed else 'LOOP'
            retained = f'ORIGINAL_{family}_RETAINED'
            row = dict(id=proof.id, proof=reference, source_loop_authority='ORIGINAL_F4_REPLAYED',
                       native_directory=None, selected_for_use=False, whole_program_compiled=False,
                       source_motion=publish_plan(folder / 'source-motion.json', motion))
            if composed:
                row.update(loop_links=spec['loop_links'], root_callee_source=asdict(proof.root_callee_source),
                           body_identity=proof.body_identity, nested_global_guard_count=len(proof.nested_callee_guards),
                           source_loop_authority='ORIGINAL_F4_COMPOSITION_AND_GLOBAL_ROUTES_REPLAYED')
            rows.append(row)
            legality = motion['movement_legality']
            if legality['status'] != 'CONDITIONAL':
                row.update(status=retained, selected_kind='NO_OP', selection_scope='SOURCE_MOVEMENT_LEGALITY',
                           source_legality_status=legality['status'],
                           reason='original source window does not justify all required invariant placements')
                if composed:
                    row['unresolved_roots'] = [root['expression'] for root in legality['roots']
                                               if root['status'] != 'CONDITIONAL']
                    row['source_blockers'] = [
                        *legality['source_window_effects']['blockers'],
                        *(dict(blocker, expression=root['expression'], property=field)
                          for root in legality['roots'] for field in ('data_available', 'no_clobber', 'execution', 'effects')
                          for blocker in root[field]['blockers']),
                    ]
                continue
            inputs = input_specs.get(proof.id, {})
            shapes = inputs.get('input_shapes', spec.get('input_shapes'))
            count = inputs.get('trip_count', trip_count)
            missing = [key for key, value in (('input_shapes', shapes), ('trip_count', count)) if value is None]
            if missing:
                row.update(status=f'{family}_INPUT_CONSTRAINTS_REQUIRED', missing=missing)
                continue
            budget = inputs.get('memory_budget_bytes', spec.get('memory_budget_bytes'))
            if budget is None:
                raise ValueError('loop requires retained incremental memory budget')
            threads = inputs.get('cpu_threads', cpu_threads)
            row.update(plan_loop_region(
                proof, hardware, folder, shapes=shapes, count=count, budget=budget, threads=threads,
                catalog=catalog if composed else None, model=cpu_model, max_threads=max_threads,
                calibrations=calibrations, original_cost=original_costs.get(proof.id),
            ))
            if row['selected_kind'] == 'REGISTER_TEMPORAL_FUSION':
                work.append((row, ScheduledLoop(proof.id, folder, reference, row['source_motion'], row['schedule'],
                                                composed, cpu_model or threads is not None)))
            else:
                row['status'] = retained
    return direct, composite, work


def _invariant_loop_links(proof,loops,reference):
    """Join original F4 identifiers, never function/source-name similarity."""
    control=next(row for row in proof.origins if row.get('role')=='single_shared_invariant_loop')
    call=next(row for row in proof.origins if row.get('role')=='invariant_array_leaf_call')
    links=[];headers=0
    for i,function in enumerate(loops['functions']):
        for j,loop in enumerate(function['loops']):
            header=loop['header'];port=header['port']
            if (header['unit']!=control['unit'] or port['template']!=control['template']
                    or port['owner']['wire']!=control['step'] or port['kind']!='ENTER'):
                continue
            headers+=1
            for k,body_call in enumerate(loop['body_calls']):
                if body_call['result']!=call['call_result']:continue
                prefix=f'#/functions/{i}/loops/{j}'
                links.append(dict(kind='EXACT_ORIGINAL_F4_LOOP_HEADER_AND_CALL_RESULT',
                    loop=dict(artifact=reference,locator=prefix),
                    call=dict(artifact=reference,locator=prefix+f'/body_calls/{k}')))
    frontier=[] if len(links)==1 else [dict(kind='ORIGINAL_LOOP_CALL_LINK',status='UNRESOLVED',
        matching_headers=headers,matching_call_links=len(links),
        reason='unique original loop header and enclosed leaf CALL result not matched')]
    return links,frontier


def _planning_sources(catalog,source):
    """Enumerate only source identities already admitted by this catalog."""
    if source is not None:return [str(Path(source).absolute())]
    store=catalog.fact_store;paths=set()
    for unit,row in store.rows.items():
        model=store.model_units.get(unit,{})
        path=model.get('path',row.get('path'))
        if type(path) is not str or not path:
            raise ValueError('admitted source unit has no retained source path: '+unit)
        paths.add(str(Path(path).absolute()))
    return sorted(paths)


def _planning_loops(view,sources,source_filter):
    from scar.analysis.source_control import source_loop_dependencies
    functions=[]
    for source in sources:
        projection=source_loop_dependencies(view,source)
        functions.extend(dict(function,source_path=source) for function in projection['functions'])
    return dict(schema='scar.source-loop-dependencies',schema_version=1,
        source=None if source_filter is None else str(source_filter),sources=sources,
        functions=functions,loop_count=sum(len(row['loops']) for row in functions),
        loop_body_call_count=sum(len(loop['body_calls']) for row in functions for loop in row['loops']),
        source_unit_scope='DECLARED_CATALOG_UNITS_NOT_RUNTIME_REACHABILITY'
            if source_filter is None else 'EXPLICIT_SOURCE_FILTER_NOT_RUNTIME_REACHABILITY',
        loops_unrolled=False,first_stage_complete=False)


@reuse_decoded_source_bodies()
def plan_build(program, out, *, source=None, shape=None, input_shapes=None,
               memory_budget_bytes, affinity=None, max_threads=None, entry_request=None):
    from scar.analysis.build import BuildSession,_check
    from scar.analysis.build_admission import publish_build_admission
    from scar.analysis.build_f4 import _validate_catalog_parents
    from scar.compiler.stateful_loop_frontend import discover_stateful_regions
    from scar.compiler.stateful_model import write_stateful
    from scar.compiler.loop_admission import discover_invariant_loops, write_invariant_loop_proof
    from scar.compiler.pipeline_loop_frontend import (discover_pipeline_invariant_loops,
        write_composite_invariant_loop_proof)
    from scar.compiler.ordered_planning_graph import retain_ordered_planning_graph
    from scar.analysis.source_control import attach_loop_call_observations
    from scar.ir.execution_recipe import artifact_reference,read_artifact
    from scar.optimization.stateful_cpu_model import build_stateful_cpu_domain
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
    source_filter=None if source is None else Path(source).absolute()
    with BuildSession(program) as session:
        target=manifest.get('target')
        if target is None:raise ValueError('program build has no target hardware snapshot')
        _check(target)
        hardware=Path(target['path']).parent
        # Reuse this session's catalog and original shared function bodies.
        sources=_planning_sources(session.view.catalog,source_filter)
        regions,rejected=discover_stateful_regions(session.view,source=source_filter)
        out.mkdir(parents=True)
        source_admission=publish_build_admission(session,out/'source-admission.json',source_filter=source_filter)
        entry_obligations=None
        if entry_request is not None:
            from scar.analysis.entry_obligations import publish_entry_obligations
            entry_obligations=publish_entry_obligations(session,out/'entry-obligations.json')
        loops=_planning_loops(session.view,sources,source_filter)
        if session._f4 is not None:
            providers=tuple(p for name,p in session._f4.providers.items() if name.startswith('bounded_call_bindings:'))
            attach_loop_call_observations(session.view,loops,providers)
        (out/'loops.json').write_text(json.dumps(loops,indent=2,sort_keys=True)+'\n')
        loop_reference=artifact_reference(out/'loops.json',loops['schema'])
        rows=[]
        for index,region in enumerate(regions):
            folder=out/('region-'+str(index));folder.mkdir()
            path=folder/'region.json';write_stateful(region,path)
            public_graph,dependencies=retain_ordered_planning_graph(session,path,folder/'program-graph')
            (folder/'dependencies').mkdir()
            (folder/'dependencies/dependencies.json').write_text(json.dumps(dependencies,
                indent=2,sort_keys=True,allow_nan=False)+'\n')
            shapes=({value.id:list(shape) for value in region.inputs if value.kind=='array'}
                if shape is not None else input_shapes.get(region.id))
            row=dict(region_id=region.id,source_code_key=region.source.code_key,
                region=str(path),source_unit_scope='DECLARED_FUNCTION_PROFILE_NOT_RUNTIME_REACHABILITY',
                dependencies=artifact_reference(folder/'dependencies/dependencies.json',dependencies['schema']),
                program_graph=public_graph,
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
        # Pure array-loop proofs share the admitted source session and original
        # hardware snapshot. The body is stored once inside its proof artifact;
        # later cost/lowering passes refer to that body instead of re-extracting
        # a disconnected array DAG or copying it per hardware configuration.
        invariant_proofs,invariant_rejected=discover_invariant_loops(session.view,source=source_filter)
        invariant_rows=[]
        for index,proof in enumerate(invariant_proofs):
            folder=out/('invariant-loop-'+str(index));folder.mkdir()
            reference=write_invariant_loop_proof(proof,folder/'proof.json')
            links,frontier=_invariant_loop_links(proof,loops,loop_reference)
            shapes=({name:list(shape) for name in proof.body.parameters} if shape is not None else None)
            invariant_rows.append(dict(id=proof.id,source_code_key=proof.source.code_key,
                status='SOURCE_LOOP_PROOF_RETAINED',proof=reference,body_pointer='#/body',
                input_shapes=shapes,trip_count_status='REQUIRES_RUNTIME_OR_DECLARED_INPUT',
                memory_budget_bytes=memory_budget_bytes,hardware=target,loop_links=links,
                loop_link_frontier=frontier,iterations_unrolled=False,
                original_f4_replay_required_for_lowering=True,
                source_replacement_authorized=False,selected_for_use=False,lowering_authorized=False))
        composite_proofs,composite_rejected=discover_pipeline_invariant_loops(session.view,source=source_filter)
        composite_rows=[]
        for index,proof in enumerate(composite_proofs):
            folder=out/('composite-loop-'+str(index));folder.mkdir()
            reference=write_composite_invariant_loop_proof(proof,folder/'proof.json')
            links,frontier=_invariant_loop_links(proof,loops,loop_reference)
            shapes=({name:list(shape) for name in proof.body.parameters} if shape is not None else None)
            composite_rows.append(dict(id=proof.id,source_code_key=proof.source.code_key,
                status='SOURCE_COMPOSITE_LOOP_PROOF_RETAINED',proof=reference,body_pointer='#/body',
                proof_kind=proof.profile,body_identity=proof.body_identity,
                nested_global_guard_count=len(proof.nested_callee_guards),
                input_shapes=shapes,trip_count_status='REQUIRES_RUNTIME_OR_DECLARED_INPUT',
                memory_budget_bytes=memory_budget_bytes,hardware=target,loop_links=links,
                loop_link_frontier=frontier,iterations_unrolled=False,
                original_f4_replay_required_for_lowering=True,runtime_global_route_guards_required=True,
                source_replacement_authorized=False,selected_for_use=False,lowering_authorized=False))
        _validate_catalog_parents(session.view.catalog)
        _check(target)
        if artifact_reference(program/'manifest.json',parent['schema'])!=parent:
            raise ValueError('program build changed during graph planning')
    report=dict(schema='scar.program-optimization-domains',schema_version=2,
        program=parent,source=str(source_filter or Path(manifest['entry']).absolute()),
        source_filter=None if source_filter is None else str(source_filter),sources=sources,
        source_unit_scope=loops['source_unit_scope'],hardware=target,regions=rows,rejections=rejected,
        source_admission=source_admission,memory_budget_bytes=memory_budget_bytes,
        input_shape_constraints=dict(common_shape=None if shape is None else list(shape),
            named_shapes=input_shapes),
        invariant_loops=invariant_rows,invariant_loop_rejections=invariant_rejected,
        composite_invariant_loops=composite_rows,composite_rejections=composite_rejected,
        semantics='ORIGINAL_F4_FUNCTION_WORLD_AND_SOURCE_PORTS',
        loop_dependencies=loop_reference,
        loops=loops['loop_count'],loop_body_calls=loops['loop_body_call_count'],loops_unrolled=False,
        catalog_reader_reused_for_region_discovery=True,
        source_ast_fallback_used=False,target_program_executed=False,
        target_variants_compiled=False,whole_program_compiled=False,
        first_stage_complete=False)
    if entry_request is not None:
        from scar.compiler.whole_entry_pipeline import prepare_whole_entry
        report['entry_obligations']=entry_obligations
        report['whole_entry']=prepare_whole_entry(report,entry_request,out)
    report['seconds']=time.monotonic()-started
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
    parser.add_argument('--entry-request',help='retained whole-entry argv and implementation evidence request')
    args=parser.parse_args(argv)
    result=plan_build(args.build,args.out,source=args.source,shape=args.shape,
        input_shapes=None if args.input_shapes is None else json.loads(Path(args.input_shapes).read_bytes()),
        memory_budget_bytes=args.memory_budget_bytes,max_threads=args.max_threads,entry_request=args.entry_request)
    print(json.dumps(dict(regions=len(result['regions']),invariant_loops=len(result['invariant_loops']),
        composite_invariant_loops=len(result['composite_invariant_loops']),
        rejections=len(result['rejections']),
        whole_entry=None if 'whole_entry' not in result else result['whole_entry']['domain']['uri'],
        result=str(Path(args.out)/'result.json'),seconds=result['seconds']),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())

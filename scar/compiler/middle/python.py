"""Derive hardware planning domains directly from an admitted program graph.

This bridge consumes original F4 operations, world ports and the build's full
hardware snapshot. Unsupported functions stay explicit; no AST fallback or
target execution supplies a replacement semantic model.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from scar.analysis.source_decoded_bodies import reuse_decoded_source_bodies
from scar.ir.execution_recipe import artifact_reference, open_artifact, read_artifact

from ..contracts import GraphBundle, ScheduledArrayLoop, ScheduledLoop
from .plan import publish_plan


def project_loop_calibrations(domain, calibrations, out):
    from scar.compiler.calibrate_temporal_cpu import project_temporal_resource_fact_batch

    out = Path(out).resolve()
    if out.exists():
        raise ValueError('fresh loop cost input publication required')
    out.mkdir(parents=True)
    facts, sources, batches, skipped, materials = [], [], [], [], []
    composite = domain['schema'] == 'scar.composite-temporal-cpu-domain'
    for value in calibrations:
        path = Path(value).resolve()
        if path.is_dir():
            path = path / 'result.json'
        reference, document = open_artifact(path)
        sources.append(reference)
        materials.append((path, document, reference))
    base_calibrations = [path for path, document, _ in materials
                         if document.get('schema') == 'scar.temporal-cpu-resource-calibration']
    for index, (path, document, reference) in enumerate(materials):
        schema = document['schema']
        folder = out / f'calibration-{index}'
        if schema == 'scar.temporal-resource-calibration':
            facts.append(path)
            continue
        if schema == 'scar.temporal-cpu-resource-calibration':
            batch = project_temporal_resource_fact_batch(domain, path, folder)
        elif schema == 'scar.composite-temporal-boundary-calibration':
            if not composite:
                skipped.append(dict(calibration=reference, reason='COMPOSITE_BOUNDARY_INAPPLICABLE_TO_DIRECT_LOOP'))
                continue
            from scar.compiler.calibrate_temporal_cpu import project_composite_boundary_facts

            batch = project_composite_boundary_facts(domain, path, folder,
                                                     base_calibrations=base_calibrations)
        else:
            raise ValueError(f'independent supported loop resource calibration required: {schema}')
        facts.extend(Path(row['observations']['uri']) for row in batch['candidates'])
        batches.append(artifact_reference(folder / 'result.json', batch['schema']))
    for reference in sources:
        read_artifact(reference, json_document=False)
    report = dict(schema='scar.loop-cost-inputs', schema_version=1, domain_id=domain['id'],
                  calibrations=sources, batches=batches, skipped=skipped,
                  facts=[artifact_reference(path, 'scar.temporal-resource-calibration') for path in facts],
                  source_independent_projections=True, target_program_executed=False)
    return facts, publish_plan(out / 'result.json', report)


def plan_loop_region(proof, hardware, directory, *, shapes, count, budget, catalog=None,
                     model=False, threads=None, max_threads=None, calibrations=(), original_cost=None):
    """Publish the schedule constructed from the admitted proof and target.

    The standalone export replays schedules from persisted evidence.
    """
    target = Path(hardware['path']).parent
    arguments = dict(input_shapes=shapes, trip_count=count, memory_budget_bytes=budget)
    report = dict(physical_optimality_proven=False, performance_qualified=False)
    if model and threads is not None:
        raise ValueError('automatic loop cost selection and an explicit CPU preview are separate modes')

    if model or threads is not None:
        from scar.optimization.temporal_cpu_choices import (
            build_temporal_cpu_domain, plan_selected_temporal_cpu_choice, preview_temporal_cpu_choice,
        )

        domain = build_temporal_cpu_domain(proof, target, **arguments, catalog=catalog,
                                           max_threads=max_threads if model else threads)
        composite = domain.get('proof_kind') == 'COMPOSITE'
        report['cpu_domain'] = publish_plan(directory / 'cpu-domain.json', domain)
        if composite:
            report['guard_profile'] = publish_plan(directory / 'guard-profile.json', domain['guard_profile'])
        if model:
            facts, report['cost_inputs'] = project_loop_calibrations(domain, calibrations, directory / 'resource-facts')
            selection = plan_selected_temporal_cpu_choice(domain, resource_facts=facts, no_op_profile=original_cost)
            choice, schedule = selection['choice'], selection['schedule']
            report.update(
                cpu_choice=publish_plan(directory / 'cpu-choice.json', choice),
                candidate_count=len(domain['candidates']),
                complete_cost_profiles=sum(row['model']['complete'] for row in choice['variants']),
                cost_domain_complete=choice['complete_cost_domain'],
                original_no_op_cost_known=choice['original_no_op']['complete'],
                selection_scope=('SOURCED_COMPOSITE_FORWARD_CPU_COMPONENT_MODEL' if composite
                                 else 'SOURCED_FORWARD_CPU_COMPONENT_MODEL'),
            )
            if schedule is None:
                report.update(selected_kind='NO_OP', cost_status=choice['status'],
                              reason='complete forward cost domain or profitable model winner unavailable')
                return report
        else:
            schedule = preview_temporal_cpu_choice(domain, cpu_threads=threads)
            report['selection_scope'] = ('EXPLICIT_COMPOSITE_CPU_TEAM_PREVIEW_UNRANKED' if composite
                                         else 'EXPLICIT_CPU_TEAM_PREVIEW_UNRANKED')
    else:
        from scar.optimization.invariant_loop_schedule import plan_invariant_loop_schedule

        schedule = plan_invariant_loop_schedule(proof, target, **arguments, catalog=catalog)
        report['selection_scope'] = schedule.get('comparison_scope', 'STRUCTURAL_LOGICAL_WORK_AND_RESOURCE_MODEL')

    snapshot = schedule['target_snapshot']
    if (snapshot['uri'], snapshot['sha256'], snapshot['size_bytes']) != (
            hardware['path'], hardware['sha256'], hardware['bytes']):
        raise ValueError('loop schedule differs from the original hardware snapshot')
    report.update(
        schedule=publish_plan(directory / 'schedule.json', schedule),
        selected_kind=schedule['selected_kind'],
        profitability_status=schedule['profitability_status'],
        input_shapes=shapes, trip_count=count,
        memory_budget_bytes=schedule['effective_incremental_budget_bytes'],
        requested_memory_budget_bytes=budget,
    )
    return report


def plan_array_loop(loop, hardware, directory, *, shapes, count, budget, model=False, threads=None):
    """Bind the existing single-carry fusion strategy to an observed CPU target."""
    from scar.compiler.model import document
    from scar.optimization.hardware import bind_cpu_placement, propose_cpu_placements
    from scar.optimization.loop_strategy import plan_loop_strategy_with_body

    if model and threads is not None:
        raise ValueError('automatic loop cost selection and an explicit CPU preview are separate modes')
    strategy, execution = plan_loop_strategy_with_body(
        loop, input_shapes=shapes, trip_count=count, memory_budget_bytes=budget)
    report = dict(
        strategy=publish_plan(directory / 'strategy.json', strategy),
        status='ORIGINAL_ARRAY_LOOP_RETAINED', selected_kind='NO_OP',
        input_shapes=strategy['input_shapes'], trip_count=strategy['trip_count'],
        memory_budget_bytes=budget, physical_optimality_proven=False, performance_qualified=False,
    )
    if strategy['selected_strategy'] == 'NO_OP':
        return dict(report, profitability_status='STRATEGY_NOT_ADMITTED',
                    reason='exact temporal fusion or its input and payload bounds are unresolved')
    if model:
        return dict(report, profitability_status='MISSING_COMPATIBLE_COMPONENT_COSTS',
                    reason='the single-carry temporal ABI has no complete comparable CPU and original costs')

    target = Path(hardware['path']).parent
    requested = 1 if threads is None else threads
    placements = propose_cpu_placements(execution, target, memory_budget_bytes=budget, max_threads=requested)
    binding = next((row for row in placements if len(row.cpu_ids) == requested
                    and row.memory_budget_bytes >= strategy['final_bytes']), None)
    if binding is None:
        return dict(report, profitability_status='NO_FEASIBLE_CPU_PLACEMENT',
                    reason='no admitted CPU cohort and memory domain fits the payload budget')
    if (binding.target_path, binding.target_sha256) != (str(target), hardware['sha256']):
        raise ValueError('array loop CPU placement differs from the original hardware snapshot')
    bound = bind_cpu_placement(execution, binding)
    return dict(report, status='ARRAY_LOOP_TEMPORAL_FUSION_SCHEDULED',
                selected_kind='ARRAY_LOOP_TEMPORAL_FUSION', profitability_status='UNMEASURED',
                selection_scope='EXPLICIT_CPU_COHORT_UNRANKED' if threads is not None
                else 'SINGLE_CPU_STRUCTURAL_CANDIDATE',
                schedule=publish_plan(directory / 'execution-plan.json', document(bound)),
                cpu_ids=list(binding.cpu_ids), memory_budget_bytes=bound.cost['memory_budget_bytes'],
                requested_memory_budget_bytes=budget)


def _prepare_loop_lowering(proof, folder, proof_ref, source_motion, schedule_ref, *, composite, parallel):
    """Freeze the selected loop's lowering inputs before handing it to code generation."""
    from scar.compiler.model import CpuPlacement

    schedule = read_artifact(schedule_ref)
    binding = dict(schedule['resource_binding'])
    binding['cpu_ids'] = tuple(binding['cpu_ids'])
    binding['evidence'] = tuple(binding.get('evidence', ()))
    placement = CpuPlacement(**binding)
    specialization = dict(input_shapes=schedule['input_shapes'], trip_count=schedule['trip_count'],
                          input_layout='C_CONTIGUOUS')
    if composite:
        from scar.compiler.invariant_loop_runtime import _loop_interface

        interface = _loop_interface(proof, execution_specialization=specialization,
                                    source_motion=source_motion)
    else:
        interface = dict(
            original_f4_proof=asdict(proof), carry_parameter=proof.carry_parameter,
            readonly_parameters=[name for name in proof.body.parameters if name != proof.carry_parameter],
            authority='ORIGINAL_F4_REPLAYED_WITH_RESOURCE_SCHEDULE',
            execution_specialization=specialization, source_motion=source_motion,
        )
    if parallel:
        from scar.compiler.temporal_invariant_loop import _parallel_plan as prepare
    else:
        from scar.compiler.temporal_invariant_loop import _plan as prepare

    plan, _ = prepare(proof.body, proof.carry_parameter, interface,
                      schedule['effective_incremental_budget_bytes'], placement)
    if plan['execution_movement_binding'] is None:
        raise ValueError('loop delivery requires consumed source movement legality')
    return publish_plan(folder / 'cpp20-lowering.json', dict(
        schema='scar.cpp20-loop-lowering', schema_version=1, loop_id=proof.id,
        proof=proof_ref, source_motion=source_motion, schedule=schedule_ref, parallel=parallel, plan=plan,
    ))


def schedule_loops(document, out, catalog, *, trip_count=None, input_specs=None, cpu_threads=None,
                   cpu_model=False, calibrations=(), original_costs=None, max_threads=None):
    """Project proven loop movement and publish schedules independently of codegen."""
    from scar.compiler.loop_admission import read_invariant_loop_motion, validate_invariant_loop_links
    from scar.compiler.pipeline_loop_frontend import (
        read_composite_invariant_loop_motion, validate_composite_invariant_loop_links,
    )

    input_specs = input_specs or {}
    original_costs = original_costs or {}
    profiles = (
        ('invariant_loops', 'loops', read_invariant_loop_motion, validate_invariant_loop_links),
        ('composite_invariant_loops', 'composite-loops', read_composite_invariant_loop_motion,
         validate_composite_invariant_loop_links),
    )
    array_specs = document.get('array_loops', ())
    known = {spec['id'] for name, *_ in profiles for spec in document.get(name, ())}
    known.update(spec['id'] for spec in array_specs)
    if (input_specs.keys() | original_costs.keys()) - known:
        raise ValueError('loop inputs and original costs must name retained loop IDs')
    direct, composite, arrays, work = [], [], [], []
    if not known:
        return direct, composite, arrays, work
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
                parallel = cpu_model or threads is not None
                row['lowering'] = _prepare_loop_lowering(
                    proof, folder, reference, row['source_motion'], row['schedule'],
                    composite=composed, parallel=parallel,
                )
                work.append((row, ScheduledLoop(proof.id, folder, reference, row['source_motion'], row['schedule'],
                                                composed, parallel, row['lowering'])))
            else:
                row['status'] = retained
    _schedule_array_loops(array_specs, document, out, catalog, arrays, work,
                          trip_count=trip_count, input_specs=input_specs,
                          cpu_threads=cpu_threads, cpu_model=cpu_model)
    return direct, composite, arrays, work


def _schedule_array_loops(specs, document, out, catalog, rows, work, *,
                          trip_count, input_specs, cpu_threads, cpu_model):
    from scar.compiler.loop_admission import read_array_loop_proof, validate_array_loop_links

    hardware = document['hardware']
    for index, spec in enumerate(specs):
        folder = out / 'array-loops' / f'loop-{index}'
        folder.mkdir(parents=True)
        reference = spec['proof']
        loop = read_array_loop_proof(reference, catalog=catalog)
        if spec['hardware'] != hardware:
            raise ValueError('array loop hardware differs from the original program domain')
        validate_array_loop_links(loop, spec, document['loop_dependencies'])
        row = dict(id=loop.id, proof=reference, loop_links=spec['loop_links'],
                   source_loop_authority='ORIGINAL_F4_LOOP_AND_PIPELINE_REPLAYED',
                   native_directory=None, selected_for_use=False, whole_program_compiled=False)
        rows.append(row)
        inputs = input_specs.get(loop.id, {})
        shapes = inputs.get('input_shapes', spec.get('input_shapes'))
        count = inputs.get('trip_count', trip_count)
        budget = inputs.get('memory_budget_bytes', spec.get('memory_budget_bytes'))
        missing = [key for key, value in (('input_shapes', shapes), ('trip_count', count),
                   ('memory_budget_bytes', budget)) if value is None]
        if missing:
            row.update(status='ARRAY_LOOP_INPUT_CONSTRAINTS_REQUIRED', missing=missing)
            continue
        row.update(plan_array_loop(loop, hardware, folder, shapes=shapes, count=count, budget=budget,
                                   model=cpu_model, threads=inputs.get('cpu_threads', cpu_threads)))
        if row['selected_kind'] == 'ARRAY_LOOP_TEMPORAL_FUSION':
            work.append((row, ScheduledArrayLoop(loop.id, folder, reference, row['strategy'],
                                                 row['schedule'], hardware)))


def _invariant_loop_links(proof,loops,reference):
    """Join original F4 identifiers, never function/source-name similarity."""
    from scar.compiler.loop_admission import loop_dependency_witnesses

    control, call = loop_dependency_witnesses(proof)
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
    from scar.compiler.graph_frontend import RegionDiscovery
    from scar.compiler.frontend.stateful import discover_stateful_regions
    from scar.compiler.stateful_model import write_stateful
    from scar.compiler.loop_admission import (document as array_loop_document, discover_graph_loops,
        discover_invariant_loops, write_invariant_loop_proof)
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
    source_filter=None if source is None else Path(source).absolute()
    with BuildSession(program) as session:
        graph=GraphBundle.from_python_session(session)
        parent=asdict(graph.evidence)
        manifest=graph.report
        target=manifest['target']
        _check(target)
        hardware=Path(graph.hardware.uri).parent
        sources=_planning_sources(session.view.catalog,source_filter)
        # Complete region admission before building per-region hardware domains.
        discovery = RegionDiscovery(session.view)
        regions,rejected=discover_stateful_regions(session.view,source=source_filter,_discovery=discovery)
        invariant_proofs,invariant_rejected=discover_invariant_loops(session.view,source=source_filter,_discovery=discovery)
        composite_proofs,composite_rejected=discover_pipeline_invariant_loops(
            session.view, source=source_filter, _discovery=discovery)
        array_proofs,array_rejected=discover_graph_loops(session.view, source=source_filter, _discovery=discovery)
        discovery.validate()
        del discovery
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
        array_rows = []
        for index, loop in enumerate(array_proofs):
            folder = out / 'array-loop-{}'.format(index)
            folder.mkdir()
            reference = publish_plan(folder / 'proof.json', array_loop_document(loop))
            links, frontier = _invariant_loop_links(loop, loops, loop_reference)
            shapes = ({loop.state_parameter: list(shape)} if shape is not None
                      else input_shapes.get(loop.id))
            array_rows.append(dict(id=loop.id, source_code_key=loop.source.code_key,
                status='SOURCE_ARRAY_LOOP_PROOF_RETAINED', proof=reference, body_pointer='#/body',
                input_shapes=shapes, trip_count_status='REQUIRES_RUNTIME_OR_DECLARED_INPUT',
                memory_budget_bytes=memory_budget_bytes, hardware=target, loop_links=links,
                loop_link_frontier=frontier, iterations_unrolled=False,
                original_f4_replay_required_for_lowering=True, source_replacement_authorized=False,
                selected_for_use=False, lowering_authorized=False))
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
        array_loops=array_rows,array_loop_rejections=array_rejected,
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
        array_loops=len(result['array_loops']),
        rejections=len(result['rejections']),
        whole_entry=None if 'whole_entry' not in result else result['whole_entry']['domain']['uri'],
        result=str(Path(args.out)/'result.json'),seconds=result['seconds']),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())

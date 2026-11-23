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
from typing import TYPE_CHECKING

from scar.analysis.source_decoded_bodies import reuse_decoded_source_bodies
from scar.ir.execution_recipe import artifact_reference, open_artifact, read_artifact

from ..contracts import ExecutionPlanRef, GraphBundle, ScheduledArrayLoop, ScheduledLoop, python_build_receipts
from .plan import publish_plan

if TYPE_CHECKING:
    from scar.compiler.loop_admission import InvariantLoopProof
    from scar.compiler.pipeline_loop_frontend import CompositeInvariantLoopProof


@dataclass(frozen=True, slots=True)
class _GraphLoopEvidence:
    proof: 'InvariantLoopProof | CompositeInvariantLoopProof'
    motion: dict
    composite: bool


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
    admitted = proof if isinstance(proof, _GraphLoopEvidence) else None
    if admitted is not None:
        proof = admitted.proof
    target = Path(hardware['path']).parent
    arguments = dict(input_shapes=shapes, trip_count=count, memory_budget_bytes=budget)
    report = dict(physical_optimality_proven=False, performance_qualified=False)
    if model and threads is not None:
        raise ValueError('automatic loop cost selection and an explicit CPU preview are separate modes')

    if model or threads is not None:
        from scar.optimization.temporal_cpu_choices import (
            _build_admitted_temporal_cpu_domain, build_temporal_cpu_domain,
            plan_selected_temporal_cpu_choice, preview_temporal_cpu_choice,
        )

        limit = max_threads if model else threads
        if admitted is None:
            domain = build_temporal_cpu_domain(proof, target, **arguments, catalog=catalog, max_threads=limit)
        else:
            domain = _build_admitted_temporal_cpu_domain(proof, target, **arguments, max_threads=limit,
                                                         composite=admitted.composite)
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
        from scar.optimization.invariant_loop_schedule import _plan_admitted_loop, plan_invariant_loop_schedule

        if admitted is None:
            schedule = plan_invariant_loop_schedule(proof, target, **arguments, catalog=catalog)
        else:
            schedule = _plan_admitted_loop(proof, target, **arguments, composite=admitted.composite)
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
        request, target_receipt = python_build_receipts(session)
        parent = asdict(request)
        manifest = session.report
        target=manifest['target']
        _check(target)
        hardware=Path(target_receipt.uri).parent
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


def _invariant_evidence(row):
    from scar.optimization.resource_solver import semantic_sha256

    proof_reference, motion_reference = row['proof'], row['source_motion']
    composite = row['kind'] == 'COMPOSITE_LOOP'
    if composite:
        from scar.compiler.pipeline_loop_frontend import _read_composite_invariant_loop_materials

        proof, _ = _read_composite_invariant_loop_materials(proof_reference['uri'],
                                                           expected_reference=proof_reference)
        family = ('scar.source-composite-loop-motion', 1)
    else:
        from scar.compiler.loop_admission import InvariantLoopProof, PROOF_SCHEMA, validate_invariant_loop_proof
        from scar.ir.record_codec import decode

        document = read_artifact(proof_reference)
        if (document.get('schema'), document.get('schema_version')) != (PROOF_SCHEMA, 1):
            raise ValueError('original F4 invariant loop proof required')
        proof = decode(InvariantLoopProof, {key: value for key, value in document.items()
                                           if key not in ('schema', 'schema_version')})
        validate_invariant_loop_proof(proof)
        family = ('scar.source-loop-motion', 2)
    motion = read_artifact(motion_reference)
    if (row['id'] != proof.id or (motion.get('schema'), motion.get('schema_version')) != family
            or motion.get('proof') != proof_reference or motion.get('proof_id') != proof.id
            or motion.get('proof_sha256') != semantic_sha256(asdict(proof))
            or motion.get('source_replacement_authorized') is not False):
        raise ValueError('source movement differs from its original F4 graph evidence')
    return _GraphLoopEvidence(proof, motion, composite)


def _loop_specialization(loop_id, parameters, shapes):
    if isinstance(shapes, (list, tuple)):
        return {name: list(shapes) for name in parameters}
    if type(shapes) is not dict:
        raise ValueError('a common shape or original input shape map is required')
    shapes = shapes.get(loop_id, shapes)
    names = set(parameters)
    if set(shapes) == names:
        return shapes
    if set(shapes) == set(parameters.values()):
        return {name: shapes[outer] for name, outer in parameters.items()}
    raise ValueError('loop shapes must name its original leaf or outer parameters')


def _payload_budget(shape, requested):
    if len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError('positive rank-two loop input shape required')
    return requested if requested is not None else 8 * shape[0] * shape[1]


def _graph_region_trial(graph, row, folder, *, shapes, count, threads, memory_budget):
    from scar.optimization.resource_solver import semantic_sha256

    hardware = dict(path=graph.hardware.uri, sha256=graph.hardware.sha256, bytes=graph.hardware.size_bytes)
    parallel = threads is not None and threads > 1
    motion_reference = row.get('source_motion')
    if row['kind'] == 'ARRAY_LOOP':
        from scar.compiler.loop_admission import check_loop_sources, loop_from_dict

        original = loop_from_dict(read_artifact(row['proof']))
        if original.id != row['id'] or motion_reference is not None:
            raise ValueError('array recurrence differs from its original graph proof')
        check_loop_sources(original)
        parameters = {original.state_parameter: original.body.parameters[0]}
        carry = original.state_parameter
    else:
        admitted = _invariant_evidence(row)
        original = admitted.proof
        if admitted.motion['movement_legality']['status'] != 'CONDITIONAL':
            return None, None, 'SOURCE_MOVEMENT_LEGALITY_UNRESOLVED'
        parameters, carry = original.parameter_map, original.carry_parameter
    if semantic_sha256(row['source']) != semantic_sha256(asdict(original.source)):
        raise ValueError('loop source differs from the original graph region')
    inputs = _loop_specialization(original.id, parameters, shapes)
    budget = _payload_budget(inputs[carry], memory_budget)
    folder.mkdir()

    if row['kind'] == 'ARRAY_LOOP':
        candidate = plan_array_loop(original, hardware, folder, shapes=inputs, count=count, budget=budget,
                                    threads=threads if parallel else None)
        if candidate['selected_kind'] != 'ARRAY_LOOP_TEMPORAL_FUSION':
            return None, None, candidate['profitability_status']
        prepared = publish_plan(folder / 'cpp20-lowering.json', dict(schema='scar.cpp20-array-loop-lowering',
            schema_version=1, loop_id=original.id, proof=row['proof'], strategy=candidate['strategy'],
            schedule=candidate['schedule'], hardware=asdict(graph.hardware)))
        profile = 'CPP20_ARRAY_LOOP_TEMPORAL_FUSION_V1'
    else:
        candidate = plan_loop_region(admitted, hardware, folder, shapes=inputs, count=count, budget=budget,
                                     threads=threads if parallel else None)
        if candidate['selected_kind'] != 'REGISTER_TEMPORAL_FUSION':
            return None, None, candidate.get('status', candidate['selected_kind'])
        prepared = _prepare_loop_lowering(original, folder, row['proof'], motion_reference, candidate['schedule'],
                                          composite=admitted.composite, parallel=parallel)
        family = 'COMPOSITE_INVARIANT_LOOP' if admitted.composite else 'INVARIANT_LOOP'
        workers = 'OPENMP' if parallel else 'SERIAL'
        profile = f'CPP20_{family}_{workers}_V1'
    return dict(operation=row['region'], option=candidate['selected_kind'], profile=profile,
                loop_id=original.id, proof=row['proof'], source_motion=motion_reference,
                schedule=candidate['schedule'], lowering=prepared), inputs, None


def _plan_python_trial(graph, region, source, shapes, delivery):
    from scar.ir.implementation_plan import (CostEstimate, ImplementationOption, PlanningDataEdge,
        PlanningOperation, PlanningProblem, PlanningResource)
    from scar.optimization.joint_planner import structural_trial_plan

    timing = CostEstimate(None, None, None, 'UNKNOWN', 'original guarded region', source,
                          ('COMPLETE_ORIGINAL_AND_NATIVE_REGION_DURATION',))
    data, resources, locations = (), (), {}
    if delivery is not None:
        schedule = read_artifact(delivery['schedule'])
        binding = schedule['resource_binding']
        host_memory = binding['memory_domain_id']
        cpus = {f'cpu:{cpu}': 1 for cpu in binding['cpu_ids']}
        placement_source = (dict(role='region_resource_schedule', artifact=delivery['schedule']),)
        resources = tuple(PlanningResource(cpu, 'EXECUTION', 1, placement_source) for cpu in cpus)
        resources += (PlanningResource(host_memory, 'MEMORY', binding['memory_budget_bytes'], placement_source),)
        locations = dict.fromkeys(shapes, host_memory)
        data = tuple(PlanningDataEdge(f'input:{index}', name, None, region, 8 * shape[0] * shape[1],
                                      host_memory, source_refs=source)
                     for index, (name, shape) in enumerate(shapes.items()))
        rows, columns = next(iter(shapes.values()))
        selected = ImplementationOption(delivery['option'], delivery['option'].lower(),
            (*source, *placement_source), cpus, {host_memory: 8 * rows * columns}, locations, {}, timing,
            {'delivery': delivery})
    original = ImplementationOption('original', 'preserve_source_region', source, {}, {}, locations, {}, timing, {})
    options = (original, selected) if delivery is not None else (original,)
    problem = PlanningProblem(graph.hardware, source, resources,
        (PlanningOperation(region, source, (), options, original.id),), data,
        assumptions=('ORIGINAL_GUARDED_CODE_RETAINED', 'NATIVE_REGION_TRIAL_HAS_UNKNOWN_DURATION'))
    return problem, structural_trial_plan(problem, delivery['option'] if delivery is not None else None)


def schedule_python_graph(graph, out, *, shapes=None, trip_count=None, threads=None, memory_budget_bytes=None,
                          entry_region=None):
    """Plan a source-qualified Python region from the published graph receipts."""
    from scar.analysis.replay_transaction import pinned_file_reference
    from scar.ir.implementation_plan import problem_document

    if (not isinstance(graph, GraphBundle) or graph.publication is None or graph.model is None
            or graph.publication.schema != 'scar.graph-bundle' or graph.evidence.schema != 'scar.python-graph-evidence'):
        raise ValueError('published Python graph bundle required')
    publication = asdict(graph.publication)
    references = graph.references()
    evidence = graph.report
    if (evidence.get('schema') != graph.evidence.schema or type(evidence.get('schema_version')) is not int
            or evidence['schema_version'] != 1
            or evidence.get('target_executed') is not False or evidence.get('loops_unrolled') is not False
            or any(evidence.get(role) != references[role] for role in ('program', 'hardware', 'request', 'calls'))):
        raise ValueError('Python graph evidence differs from its published source and target bindings')
    for receipt in (graph.publication, graph.request, graph.evidence):
        if pinned_file_reference(receipt.uri, receipt.schema) != asdict(receipt):
            raise ValueError('published Python graph input changed before planning')
    if threads is not None and (type(threads) is not int or threads < 1):
        raise ValueError('positive explicit CPU thread count required')
    if entry_region is not None and entry_region not in graph.entry_regions:
        raise ValueError('selected region must name an original graph entry')
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('fresh Python planning output required')
    out.mkdir(parents=True)
    source = (dict(role='original_program', artifact=references['program']),
              dict(role='source_evidence', artifact=references['graph_evidence']))
    region_ids = {region.id.wire for region in graph.model.regions}
    entry = region = entry_region if entry_region is not None else graph.entry_regions[0]
    frontiers, specialization, delivery = [], None, None
    for index, row in enumerate(evidence['regions']):
        if row['region'] not in region_ids:
            raise ValueError('Python proof region is absent from the original program graph')
        if entry_region is not None and row['region'] != entry_region:
            continue
        if row['kind'] not in ('INVARIANT_LOOP', 'COMPOSITE_LOOP', 'ARRAY_LOOP'):
            frontiers.append(dict(region=row['region'], kind=row['kind'], reason='LOWERING_PROFILE_OPEN'))
            continue
        if delivery is not None:
            frontiers.append(dict(region=row['region'], kind=row['kind'], reason='SEPARATE_REGION_TRIAL_REQUIRED'))
            continue
        if shapes is None or trip_count is None:
            frontiers.append(dict(region=row['region'], kind=row['kind'], reason='INPUT_SHAPES_OR_TRIP_COUNT_REQUIRED'))
            continue
        if evidence.get('source_filter') is not None and row['source']['path'] != evidence['source_filter']:
            raise ValueError('loop proof is outside the original source graph entry scope')
        delivery, specialization, reason = _graph_region_trial(graph, row, out / f'region-{index}',
            shapes=shapes, count=trip_count, threads=threads, memory_budget=memory_budget_bytes)
        if delivery is None:
            frontiers.append(dict(region=row['region'], kind=row['kind'], reason=reason))
            continue
        source = (*source, dict(role='original_region_proof', artifact=row['proof']))
        if delivery['source_motion'] is not None:
            source = (*source, dict(role='source_motion', artifact=delivery['source_motion']))
        region = row['region']
        if region in graph.entry_regions:
            entry = region

    problem, plan = _plan_python_trial(graph, region, source, specialization, delivery)
    deliveries = {} if delivery is None else {region: delivery}
    lowering_reference = publish_plan(out / 'lowering.json', dict(schema='scar.python-cpp20-lowering', schema_version=1,
                                                                 graph=publication, operations=deliveries))
    artifacts = dict(references, planning_problem=publish_plan(out / 'planning-problem.json', problem_document(problem)),
                     execution_plan=publish_plan(out / 'plan.json', plan))
    decision = dict(admitted=False, trial_only=bool(deliveries), default_backend='NO_OP', frontiers=frontiers,
                    reason='UNPRICED_STRUCTURAL_REGION_TRIAL' if deliveries else 'NO_SOURCE_QUALIFIED_REGION_TRIAL',
                    timing_qualified=False, deployment_authorized=False)
    return ExecutionPlanRef(entry, artifacts, decision, 'PYTHON_STRUCTURAL_TRIAL_ONLY',
                            graph=publication, lowering=lowering_reference)


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

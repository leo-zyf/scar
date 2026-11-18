"""Join admitted source graphs, controlled resources and existing native stages.

No target execution or new code generation occurs here. Strict whole-entry
and controlled-scope plans remain separate. A declared timing offer is useful
for conditional analysis without becoming a qualified deployment decision.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import CostEstimate, problem_document, digest, validate_problem
from scar.ir.record_codec import decode
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.controlled-region-preparation'
_PRODUCERS = ('compiler/plan_controlled_region.py', 'compiler/controlled_resources.py',
    'analysis/stateful_semantic_graph.py', 'optimization/controlled_region_planning.py',
    'optimization/stateful_entry_cost.py', 'optimization/region_implementation.py',
    'optimization/constructive_planner.py', 'optimization/joint_planner.py',
    'analysis/replay_transaction.py', 'analysis/target_graph.py',
    'optimization/hardware.py', 'ir/record_batch.py', 'optimization/entry_cost_use.py',
    'compiler/collect_native_optimization.py', 'compiler/native_lowering_view.py',
    'analysis/gcc_optimization.py', 'analysis/gcc_cost_features.py',
    'analysis/bounded_entry_calls.py', 'codegen/native_stage_group.py',
    'optimization/native_stage_context.py', 'compiler/calibrate_native_groups.py',
    'optimization/cost_context_compatibility.py', 'analysis/source_function_summary.py',
    'analysis/interprocedural_summary.py', 'compiler/source_summary_binding.py',
    'compiler/admission_context.py', 'compiler/stateful_candidate.py',
    'analysis/stateful_region_effects.py', 'analysis/stateful_dependencies.py',
    'analysis/source_fact_materials.py',
    'optimization/stateful_schedule.py', 'compiler/stateful_model.py',
    'analysis/controlled_native_time.py', 'optimization/controlled_native_tasks.py',
    'optimization/native_group_costs.py', 'analysis/schedule_time.py',
    'ir/versioned_schedule.py', 'ir/implementation_plan.py',
    'analysis/stateful_entry_protocol.py', 'compiler/canonical_stateful_inputs.py',
    'compiler/calibrate_stateful_entry.py')


def _pins():
    root = Path(__file__).resolve().parents[1]
    return {name:sha256((root/name).read_bytes()).hexdigest() for name in _PRODUCERS}


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    return artifact_reference(path, value['schema'])


def _reference(value, schema=None):
    if type(value) is dict:
        return value
    path = Path(value).resolve()
    return artifact_reference(path, schema or json.loads(path.read_bytes())['schema'])


def _stateful_stage_binding(region, graph, contract):
    """Use the existing source->stage maps, not names or reconstructed math."""
    stage_ref = contract['identity']['stage_plan']; plan = read_artifact(stage_ref)
    if (plan['region_id'] != region.id or plan['region_sha256'] != graph['region_sha256']
            or [row['id'] for row in plan['source_operations']] != [row.id for row in region.actions]):
        raise ValueError('actual native stage plan differs from the original source action sequence')
    actions = {row['action']:row['operation'] for row in graph['projection']['actions']}
    expressions = {row['expression']:row['operation'] for row in graph['projection']['expressions']}
    stages = {row['id']:row for row in plan['stages']}
    if (set(plan['expression_stages']) != set(expressions)
            or any(set(ids)-set(stages) for ids in plan['expression_stages'].values())):
        raise ValueError('native expression-stage map lost source expressions or references absent stages')
    rows = []
    for index, stage in enumerate(plan['stages']):
        if set(stage['source_action_ids'])-set(actions) or set(stage['expr_ids'])-set(expressions):
            raise ValueError('native stage names an unmapped original action or expression')
        if stage['kind'] == 'COMMIT':
            execution = dict(policy='RESTORED_CALLER_AFFINITY_NO_CORE_ASSIGNMENT',
                caller_lane=deepcopy(contract['execution']['caller_lane']))
        else:
            execution = deepcopy(contract['execution']['affinity_stage'])
        rows.append(dict(stage=stage['id'], index=index, kind=stage['kind'],
            source_operations=[actions[key] for key in stage['source_action_ids']],
            source_expressions=[expressions[key] for key in stage['expr_ids']],
            predecessors=deepcopy(stage['depends_on']), root_expression=stage['root_expr'],
            input_payloads=deepcopy(stage['input_payloads']), output_payload=stage['output_payload'],
            hardware_execution=execution, duration_ns=None,
            source_ref=dict(artifact=stage_ref, locator='#/stages/'+str(index))))
    covered={key for stage in plan['stages'] for key in stage['source_action_ids']}
    if covered != set(actions):
        raise ValueError('some source execution actions have no native stage correspondence')
    return dict(schema='scar.stateful-source-native-correspondence', schema_version=1,
        semantic_graph_revision=graph['graph_revision'], semantic_root=graph['root'],
        region_id=region.id, region_sha256=graph['region_sha256'], stage_plan=stage_ref,
        stage_plan_sha256=semantic_sha256(plan), stages=rows,
        expression_stages={key:dict(operation=expressions[key], stages=value)
            for key,value in plan['expression_stages'].items()},
        private_assignment=deepcopy(plan['assignment']), private_slots=deepcopy(contract['memory']['private_slots']),
        commits=deepcopy(plan['commit_rows']), source_actions_covered=len(covered),
        original_write_can_map_to_private_write_and_external_commit=True,
        source_actions_are_not_counted_twice=True, storage_allocation_ids_created=False,
        source_order_edges_are_not_native_stage_order=True,
        source_to_native_order_legality='EXISTING_GUARDED_STAGE_PLAN_AND_SOURCE_REPLAY_NOT_EDGEWISE_ORDER_IDENTITY',
        stage_cpu_assignment_is_not_whole_entry_assignment=True,
        timing_qualified=False, runtime_execution_observed_now=False, deployment_authorized=False)


def _compiler_annotations(reference, contract, graph, correspondence):
    from scar.compiler.collect_native_optimization import check_gcc_optimization, read_gcc_feedback
    from scar.analysis.gcc_cost_features import project_gcc_vector_costs
    feedback = check_gcc_optimization(reference)
    subject = feedback['subject']
    if type(subject) is not dict or subject.get('kind') != 'STATEFUL_NATIVE_LOWERING':
        raise ValueError('source-bound actual stateful lowering feedback required')
    view = subject['view']
    if (view['native_manifest'] != contract['implementation']['native_manifest']
            or view['identity']['build_id'] != contract['implementation']['build_id']
            or view['identity']['region_sha256'] != graph['region_sha256']):
        raise ValueError('compiler feedback belongs to another admitted native/source region')
    projection = read_gcc_feedback(feedback['files']['feedback'])
    stage_by_index = {row['index']:row for row in correspondence['stages']}
    rows = []
    for stage in view['stages']:
        bound = stage_by_index[stage['index']]
        if bound['stage'] != stage['id'] or bound['kind'] != stage['kind']:
            raise ValueError('compiler stage map differs from admitted source-native correspondence')
        span_indices = {index for index, span in enumerate(projection['stage_spans'])
            if span['id'] == 'stage:'+str(stage['index'])}
        diagnostics = [index
            for index, diagnostic in enumerate(projection['diagnostics'])
            if span_indices.intersection(diagnostic['stage_span_indices'])]
        rows.append(dict(stage=stage['id'], index=stage['index'], source_operations=bound['source_operations'],
            source_expressions=bound['source_expressions'], diagnostic_indices=diagnostics,
            association='DIAGNOSTIC_CITES_STAGE_SOURCE_NOT_EXECUTION_OR_REWRITE_PROOF',
            coverage='CITED' if diagnostics else 'NO_CITATION_NOT_DEAD_WORK_PROOF'))
    return dict(schema='scar.source-native-compiler-annotations', schema_version=1,
        semantic_graph_revision=graph['graph_revision'], compiler_collection=reference,
        native_manifest=view['native_manifest'], compiler_record=feedback['files']['record'],
        projection=feedback['files']['feedback'], stages=rows,
        compiler_cost_features=project_gcc_vector_costs(reference),
        diagnostic_index_base='ZERO_BASED_IN_SHARED_PROJECTION_DIAGNOSTICS',
        omitted_stage_kinds=view['omitted_stage_kinds'], duration_model_derived=False,
        compiler_counts_are_runtime_observations=False, timing_qualification_granted=False,
        resource_obligations_discharged=False, original_semantic_dependencies_rewritten=False)


def _entry_origin_document(reference, witness, source):
    if (witness['witnessed'] is not True or witness['requested_source']!=dict(
            path=str(Path(source.path).absolute()),sha256=source.sha256,code_key=source.code_key)):
        raise ValueError('native source code has no witnessed conditional entry link in this projection')
    return dict(schema='scar.controlled-entry-source-origin', schema_version=1,
        entry_projection=reference, witness=witness,
        proves_source_code_origin_only=True, whole_entry_legality_or_state_identity_proven=False)


def _entry_origin(reference, region, *, _context=None):
    from scar.analysis.build import BuildSession
    from scar.analysis.bounded_entry_calls import derive_entry_source_witness
    projection = read_artifact(reference)
    def derive(session):
        source = region.source
        return derive_entry_source_witness(session, projection,
            path=source.path, sha256=source.sha256, code_key=source.code_key)
    if _context is None:
        with BuildSession(Path(projection['program']['uri']).parent) as session:
            witness=derive(session)
    else:
        witness=derive(_context.program(projection['program']))
    return _entry_origin_document(reference,witness,region.source)


def _group_context(reference, contract, correspondence, compiler_feedback):
    from scar.compiler.calibrate_native_groups import check_native_group_observations
    from scar.optimization.cost_context_compatibility import compare_native_group_compilation
    observations = check_native_group_observations(reference)
    design = read_artifact(observations['design']); view = design['observer']['parent_view']
    if view['native_manifest'] != contract['implementation']['native_manifest']:
        raise ValueError('native group observations belong to another admitted native implementation')
    stages = {row['index']:row for row in correspondence['stages']}
    groups = [dict(index=group['index'], stage_indices=group['stage_indices'],
        source_stages=[dict(stage=stages[index]['stage'],
            source_operations=stages[index]['source_operations'],
            source_expressions=stages[index]['source_expressions']) for index in group['stage_indices']],
        begin_cut=group['begin_cut'], end_cut=group['end_cut']) for group in observations['group_layout']]
    report = dict(schema='scar.controlled-native-group-context', schema_version=1,
        observations=reference, design=observations['design'],
        stage_context=dict(artifact=observations['design'], locator='#/stage_context'),
        native_manifest=view['native_manifest'], source_group_map=groups,
        compiler_compatibility=None if compiler_feedback is None else
            compare_native_group_compilation(compiler_feedback, reference),
        statistics=dict(artifact=reference, locator='#/statistics'),
        scope=design['timer'], retained_transfer_residuals=observations['residuals'],
        component_mean_is_not_whole_method_cost=True,
        complete_call_cost_inferred=False, cost_use_qualification_granted=False,
        resource_obligations_discharged=False, target_samples_refitted=False)
    return report, observations, design


def _native_task_expansion(controlled, contract, correspondence, context, observations, design):
    """Refine the actual native recipe, without distributing whole-call cost.

    The strict method remains a parent boundary.  This child problem consumes
    original tasks and storage dependencies; it does not grant permission to
    reorder an ELF which still executes its original stage sequence.
    """
    from scar.optimization.controlled_native_tasks import project_controlled_native_tasks
    from scar.optimization.native_group_costs import derive_native_group_costs
    costs = None if observations is None else derive_native_group_costs(observations, design)
    groups = () if costs is None else costs['cost_groups']
    problem, projection = project_controlled_native_tasks(controlled, contract,
        correspondence, context, cost_groups=groups)
    return problem_document(problem), projection, costs


def _summary_native_correspondence(reference, binding, region, graph, correspondence):
    """Join exact original F4 ports to semantic operations and native stages.

    These are source associations, not runtime storage/version equalities.
    Shared evaluate/Expr projections keep the same semantic operation ID.
    """
    summary = read_artifact(reference)
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    def step_key(point):
        if not isinstance(point, dict) or 'port' not in point:
            return None
        port = point['port']
        return point['unit'], canonical(port['template']), port['owner']['wire']
    operations, values, accesses = {}, {}, {}
    for index, function in enumerate(summary['functions']):
        local = function['local']; base = '#/functions/'+str(index)+'/local/'
        for ordinal, operation in enumerate(local['operation_classifications']):
            operations.setdefault(step_key(operation['entry']), []).append(dict(
                artifact=reference, locator=base+'operation_classifications/'+str(ordinal)))
        for ordinal, dependency in enumerate(local['port_dependencies']):
            values.setdefault(canonical(dependency['port']), []).append(dict(
                artifact=reference, locator=base+'port_dependencies/'+str(ordinal)))
        for table in ('reads', 'writes'):
            for ordinal, access in enumerate(local[table]):
                # A receiver or RHS producer is not the access operation.
                key = step_key(access.get('result_world'))
                if key is not None:
                    accesses.setdefault(key, []).append(dict(artifact=reference,
                        locator=base+table+'/'+str(ordinal)))
    def unique(rows):
        return [row for _, row in sorted({canonical(row):row for row in rows}.items())]
    source_region = binding['region']; rows = []
    expression_origins = {}
    for index, origin in enumerate(region.source.origins):
        if origin.get('expression') is not None:
            expression_origins.setdefault(origin['expression'], []).append((origin,
                dict(artifact=source_region, locator='#/source/origins/'+str(index))))
    for kind, projected in (('EXPRESSION', graph['projection']['expressions']),
                            ('ACTION', graph['projection']['actions'])):
        for index, projection in enumerate(projected):
            identifier = projection['expression' if kind=='EXPRESSION' else 'action']
            if kind=='ACTION' and region.actions[index].id!=identifier:
                raise ValueError('summary source association lost original action order')
            origins = (expression_origins.get(identifier, []) if kind=='EXPRESSION' else
                [(origin, dict(artifact=source_region, locator='#/actions/'+str(index)+'/origins/'+str(i)))
                    for i,origin in enumerate(region.actions[index].origins)])
            op_refs, value_refs, access_refs, origin_refs = [], [], [], []
            unresolved_results = 0
            for origin, origin_ref in origins:
                if origin.get('kind')!='original_f4' or not all(key in origin for key in ('unit','template','step')):
                    continue
                key = origin['unit'], canonical(origin['template']), origin['step']
                op_refs.extend(operations.get(key, [])); access_refs.extend(accesses.get(key, []))
                origin_refs.append(origin_ref)
                for result in origin.get('results', []):
                    matched = values.get(canonical(result), [])
                    value_refs.extend(matched); unresolved_results += int(not matched)
            # Correspondence stores semantic IDs; source IDs are joined
            # through the already admitted projection, never by source names.
            stages = [stage['stage'] for stage in correspondence['stages']
                if projection['operation'] in stage['source_expressions' if kind=='EXPRESSION' else 'source_operations']]
            rows.append(dict(kind=kind, source_id=identifier, semantic_operation=projection['operation'],
                native_stages=stages, original_origins=unique(origin_refs),
                source_operations=unique(op_refs), result_dependencies=unique(value_refs),
                explicit_accesses=unique(access_refs), unassociated_result_ports=unresolved_results,
                status='MAPPED_ORIGINAL_SOURCE_MAY_FACTS' if op_refs or value_refs else 'UNRESOLVED_SOURCE_ASSOCIATION'))
    return dict(schema='scar.source-summary-native-correspondence', schema_version=1,
        summary=reference, region=source_region, region_sha256=graph['region_sha256'],
        semantic_graph_revision=graph['graph_revision'], native_manifest=binding['native_manifest'],
        entries=rows, mapped_entries=sum(row['status']=='MAPPED_ORIGINAL_SOURCE_MAY_FACTS' for row in rows),
        source_association_only=True, normal_completion_projection_only=True,
        action_and_expression_rows_are_shared_operation_projections=True,
        runtime_value_or_storage_identity_proven=False, effects_closed=False,
        transformation_authorized=False, cost_or_timing_inferred=False)


def _region_effect_binding(region, input_shapes, reference, graph, correspondence):
    from scar.analysis.stateful_region_effects import derive_stateful_region_effects
    analysis=derive_stateful_region_effects(region,input_shapes=input_shapes)
    actions={row['action']:row['operation'] for row in graph['projection']['actions']}
    expressions={row['expression']:row['operation'] for row in graph['projection']['expressions']}
    if (analysis['region_sha256']!=graph['region_sha256'] or
            [row['id'] for row in analysis['actions']]!=[row.id for row in region.actions] or
            set(expressions)!={row['id'] for row in analysis['expressions']}):
        raise ValueError('region effects differ from the original source graph')
    positions={row.id:index for index,row in enumerate(region.actions)}
    dependencies=[]
    for index, dependency in enumerate(analysis['dependencies']):
        before,after=dependency['before'],dependency['after']
        if before not in actions or after not in actions or positions[before]>=positions[after]:
            raise ValueError('effect dependency contradicts retained original action order')
        dependencies.append(dict(effect_dependency='#/analysis/dependencies/'+str(index),
            kind=dependency['kind'],before=actions[before],after=actions[after]))
    bindings=[]
    for table,mapping in (('actions',actions),('expressions',expressions)):
        for index,row in enumerate(analysis[table]):
            operation=mapping[row['id']]
            bindings.append(dict(effect='#/analysis/'+table+'/'+str(index),semantic_operation=operation,
                native_stages=[stage['stage'] for stage in correspondence['stages']
                    if operation in stage['source_operations' if table=='actions' else 'source_expressions']]))
    return dict(schema='scar.source-region-effect-binding',schema_version=1,region=reference,
        semantic_graph_revision=graph['graph_revision'],analysis=analysis,bindings=bindings,
        semantic_dependencies=dependencies,original_source_order_retained=True,
        source_association_is_not_physical_alias_proof=True,
        runtime_guards_discharged=False,effects_closed_for_whole_python_function=False,
        transformation_or_parallelization_authorized=False,cost_or_timing_inferred=False)


def _derive(kind, admission, profiles, *, cost_use='EXPLICIT_DECLARED_ANALYSIS', cost_qualification=None,
            compiler_feedback=None, entry_calls=None, native_group_observations=None, source_summary=None):
    from scar.analysis.replay_transaction import replay_transaction
    # A prepare and its independent check deliberately start separate scopes.
    # No cached validation document crosses this source-admission boundary.
    with replay_transaction():
        return _derive_body(kind, admission, profiles, cost_use=cost_use, cost_qualification=cost_qualification,
            compiler_feedback=compiler_feedback, entry_calls=entry_calls,
            native_group_observations=native_group_observations, source_summary=source_summary)


def _derive_body(kind, admission, profiles, *, cost_use='EXPLICIT_DECLARED_ANALYSIS', cost_qualification=None,
                 compiler_feedback=None, entry_calls=None, native_group_observations=None, source_summary=None):
    from scar.compiler.admission_context import AdmissionContext
    with AdmissionContext() as context:
        return _derive_in_context(kind, admission, profiles, cost_use=cost_use,
            cost_qualification=cost_qualification, compiler_feedback=compiler_feedback,
            entry_calls=entry_calls, native_group_observations=native_group_observations,
            source_summary=source_summary, _context=context)


def _derive_in_context(kind, admission, profiles, *, cost_use, cost_qualification,
                      compiler_feedback, entry_calls, native_group_observations, source_summary, _context):
    from scar.compiler.controlled_resources import derive_controlled_resource_contract
    from scar.optimization.controlled_region_planning import project_controlled_region
    contract = derive_controlled_resource_contract(kind, admission, _context=_context)
    entry = correspondence = entry_use = annotations = origin = group_context = summary_binding = summary_correspondence = effects = None
    native_context = observations = group_design = None
    if cost_use not in ('EXPLICIT_DECLARED_ANALYSIS', 'QUALIFIED_TARGET_TIMING'):
        raise ValueError('explicit cost-use contract required')
    if kind == 'STATEFUL_SPECIALIZATION':
        from scar.compiler.stateful_model import stateful_from_dict
        from scar.analysis.stateful_semantic_graph import derive_stateful_semantic_graph
        from scar.optimization.stateful_entry_cost import derive_stateful_entry_cost
        from scar.optimization.region_implementation import project_stateful_entry_implementation
        spec = read_artifact(contract['parent_admission'])
        region = stateful_from_dict(read_artifact(spec['source_region']))
        refs = [dict(role=key.upper(), artifact=spec[key]) for key in
            ('program', 'source_region', 'domain', 'candidate', 'inputs')]
        graph = derive_stateful_semantic_graph(region, input_shapes=spec['input_shapes'],
            source_refs=refs, source_proof=contract['cold_admission']['report'])
        correspondence = _stateful_stage_binding(region, graph, contract)
        effects=_region_effect_binding(region,spec['input_shapes'],spec['source_region'],graph,correspondence)
        if source_summary is not None:
            if entry_calls is None:
                raise ValueError('source summary binding requires its original entry CALL projection')
            from scar.compiler.source_summary_binding import derive_stateful_source_summary_binding
            summary_binding = derive_stateful_source_summary_binding(source_summary, admission,
                entry_calls=entry_calls, _context=_context)
            origin = _entry_origin_document(entry_calls,summary_binding['entry_witness'],region.source)
            summary_correspondence = _summary_native_correspondence(source_summary,
                summary_binding, region, graph, correspondence)
        elif entry_calls is not None:
            origin = _entry_origin(entry_calls, region, _context=_context)
        if compiler_feedback is not None:
            annotations = _compiler_annotations(compiler_feedback, contract, graph, correspondence)
        if native_group_observations is not None:
            group_context, observations, group_design = _group_context(
                native_group_observations, contract, correspondence, compiler_feedback)
            native_context = group_design['stage_context']
        identity = contract['identity']
        offers = tuple(dict(artifact=reference, locator='#') for reference in profiles)
        entry = derive_stateful_entry_cost(identity['domain']['uri'], identity['candidate_id'], offers)
        strict = project_stateful_entry_implementation(identity['domain']['uri'], identity['candidate_id'],
            entry, profiles=offers)
        from scar.optimization.entry_cost_use import admit_entry_cost_for_use
        from scar.analysis.stateful_entry_protocol import derive_stateful_entry_protocol
        requested_protocol = derive_stateful_entry_protocol(
            contract['implementation']['native_manifest'], trial=True,
            exclusive_inputs=True, selection=None)
        entry_use = admit_entry_cost_for_use(entry, use=cost_use,
            qualification_ref=cost_qualification,
            native_manifest_ref=contract['implementation']['native_manifest'],
            requested_entry_protocol=requested_protocol)
        operation = strict.operations[0]
        strict = replace(strict, operations=(replace(operation, options=(replace(operation.options[0],
            cost=decode(CostEstimate, entry_use['active_cost'])),)),))
        graph_binding = dict(schema=graph['schema'], semantic_sha256=semantic_sha256(graph),
            graph_revision=graph['graph_revision'], root=graph['root'],
            execution_members=deepcopy(graph['projection']['execution_members']),
            correspondence_sha256=semantic_sha256(correspondence))
    elif kind == 'PIPELINE_FUSION':
        if (cost_use != 'EXPLICIT_DECLARED_ANALYSIS' or cost_qualification is not None
                or compiler_feedback is not None or entry_calls is not None or native_group_observations is not None
                or source_summary is not None):
            raise ValueError('qualified entry-cost use currently requires a stateful source admission')
        from scar.optimization.region_implementation import project_region_implementation
        if profiles:
            raise ValueError('fusion costs must retain the source macro preparation profile')
        build = read_artifact(contract['parent_admission']); preparation = read_artifact(build['preparation'])
        parent = read_artifact(preparation['macro_preparation'])
        region, blueprint, report = (read_artifact(parent['files'][key]) for key in ('region', 'blueprint', 'pass_report'))
        revision = read_artifact(parent['files']['semantic_revision'])
        graph = read_artifact(revision['files']['after'])
        graph_binding = dict(schema=graph['schema'], semantic_sha256=semantic_sha256(graph),
            graph_revision=revision['after_revision'], root=revision['root'],
            execution_members=[dict(kind='operation_definition', value=revision['mapping']['added_definition']['value'],
                wire=revision['mapping']['added_definition']['wire'])])
        strict = project_region_implementation(region, blueprint, report)
    else:
        raise ValueError('known source-admitted resource kind required')
    operation = strict.operations[0]; mode = operation.options[0]
    parameters = deepcopy(mode.parameters); parameters['source_semantic_graph'] = graph_binding
    if entry_use is not None:
        parameters['cost_use'] = dict(use=entry_use['use'], report_sha256=semantic_sha256(entry_use),
            timing_qualified=entry_use['timing_qualified'],
            qualification_reference=entry_use['qualification_reference'],
            native_manifest=contract['implementation']['native_manifest'],
            requested_entry_protocol=deepcopy(requested_protocol))
    if annotations is not None:
        parameters['compiler_annotations'] = dict(semantic_sha256=semantic_sha256(annotations),
            compiler_collection=compiler_feedback, timing_qualification_granted=False)
    if origin is not None:
        parameters['entry_source_origin'] = dict(entry_projection=entry_calls,
            semantic_sha256=semantic_sha256(origin), proves_source_code_origin_only=True)
    if group_context is not None:
        parameters['native_group_context'] = dict(observations=native_group_observations,
            semantic_sha256=semantic_sha256(group_context), complete_call_cost_inferred=False)
    if summary_binding is not None:
        parameters['source_function_summary'] = dict(summary=source_summary,
            semantic_sha256=semantic_sha256(summary_binding), conditional_may_dependencies_only=True,
            correspondence_sha256=semantic_sha256(summary_correspondence),
            source_effect_or_timing_qualification_granted=False)
    if effects is not None:
        parameters['source_region_effects']=dict(semantic_sha256=semantic_sha256(effects),
            analysis_id=effects['analysis']['id'],physical_alias_proof=False,
            source_order_retained=True,runtime_guards_discharged=False)
    strict = validate_problem(replace(strict, operations=(replace(operation,
        id=graph_binding['root']['wire'], source_refs=operation.source_refs+(graph_binding,),
        options=(replace(mode, parameters=parameters),)),), source_refs=strict.source_refs+(graph_binding,)))
    controlled, projection = project_controlled_region(strict, contract)
    native_problem = native_projection = native_costs = None
    if kind == 'STATEFUL_SPECIALIZATION':
        if native_context is None:
            from scar.optimization.native_stage_context import derive_native_stage_context
            native_context = derive_native_stage_context(contract['implementation']['native_manifest'])
        native_problem, native_projection, native_costs = _native_task_expansion(controlled, contract,
            correspondence, native_context, observations, group_design)
    return dict(contract=contract, source_graph=graph, correspondence=correspondence,
        entry_cost=entry, entry_cost_use=entry_use, compiler_annotations=annotations,
        entry_source_origin=origin, native_group_context=group_context, source_summary_binding=summary_binding,
        source_summary_correspondence=summary_correspondence,
        source_region_effects=effects,
        native_task_problem=native_problem, native_task_projection=native_projection,
        native_task_costs=native_costs,
        strict_problem=problem_document(strict), controlled_problem=problem_document(controlled),
        projection=projection), strict, controlled


def _without_elapsed(value):
    return {key:row for key,row in value.items() if key != 'elapsed_seconds'}


def _metadata(kind, reference, profiles, max_seconds, files, strict, controlled, strict_plan, plan,
              cost_use, cost_qualification, compiler_feedback, entry_calls, native_group_observations, source_summary):
    return dict(schema=SCHEMA, schema_version=4, kind=kind, admission=reference,
        entry_profiles=list(profiles), max_seconds=max_seconds, files=files, producer_pins=_pins(),
        cost_use=cost_use, cost_qualification=cost_qualification,
        compiler_feedback=compiler_feedback,
        entry_calls=entry_calls, native_group_observations=native_group_observations, source_summary=source_summary,
        strict_status=strict_plan['status'], controlled_status=plan['status'],
        strict_problem_sha256=digest(strict), controlled_problem_sha256=digest(controlled),
        controlled_feasibility_is_not_whole_process_feasibility=True,
        native_task_expansion_available=files.get('native_task_problem') is not None,
        native_task_expansion_authorizes_reordering=False,
        cost_profiles_are_explicit_analysis_offers=True, timing_qualified=False,
        original_whole_entry_obligations_discharged=False, target_program_executed=False,
        native_loaded=False, compiled=False, deployment_authorized=False, default_backend='NO_OP')


def prepare(kind, admission, out, *, entry_profiles=(), max_seconds=30,
            cost_use='EXPLICIT_DECLARED_ANALYSIS', cost_qualification=None, compiler_feedback=None,
            entry_calls=None, native_group_observations=None, source_summary=None):
    from scar.optimization.constructive_planner import construct_implementations
    from scar.optimization.joint_planner import verify_execution_plan
    reference = _reference(admission)
    profiles = tuple(_reference(path) for path in entry_profiles)
    qualification = None if cost_qualification is None else _reference(cost_qualification)
    feedback = None if compiler_feedback is None else _reference(compiler_feedback)
    calls = None if entry_calls is None else _reference(entry_calls)
    observed = None if native_group_observations is None else _reference(native_group_observations)
    summary = None if source_summary is None else _reference(source_summary)
    directory = Path(out).resolve()
    if directory.exists():
        raise ValueError('fresh controlled region output required')
    documents, strict, controlled = _derive(kind, reference, profiles,
        cost_use=cost_use, cost_qualification=qualification, compiler_feedback=feedback,
        entry_calls=calls, native_group_observations=observed, source_summary=summary)
    strict_plan = construct_implementations(strict, max_seconds=max_seconds)
    plan = construct_implementations(controlled, max_seconds=max_seconds)
    verification = verify_execution_plan(controlled, plan) if plan['selected'] else None
    documents.update(strict_plan=strict_plan, controlled_plan=plan)
    if documents['native_task_problem'] is not None:
        from scar.ir.implementation_plan import problem_from_dict
        internal = problem_from_dict(documents['native_task_problem'])
        internal_plan = construct_implementations(internal, max_seconds=max_seconds)
        documents['native_task_plan'] = internal_plan
        if internal_plan['selected']:
            documents['native_task_schedule_verification'] = dict(
                schema='scar.controlled-native-task-schedule-verification', schema_version=1,
                **verify_execution_plan(internal, internal_plan))
    if verification is not None:
        documents['schedule_verification'] = dict(schema='scar.controlled-schedule-verification', schema_version=1, **verification)
    directory.mkdir(parents=True)
    files = {key:None if value is None else _write(directory/(key.replace('_','-')+'.json'), value)
        for key,value in documents.items()}
    report = _metadata(kind, reference, profiles, max_seconds, files, strict, controlled, strict_plan, plan,
        cost_use, qualification, feedback, calls, observed, summary)
    _write(directory/'preparation.json', report)
    return report


def check(preparation):
    reference = _reference(preparation, SCHEMA); saved = read_artifact(reference)
    if (saved.get('schema'), saved.get('schema_version')) != (SCHEMA, 4) or saved['producer_pins'] != _pins():
        raise ValueError('controlled region producer or schema differs')
    documents, strict, controlled = _derive(saved['kind'], saved['admission'], saved['entry_profiles'],
        cost_use=saved['cost_use'], cost_qualification=saved['cost_qualification'],
        compiler_feedback=saved['compiler_feedback'], entry_calls=saved['entry_calls'],
        native_group_observations=saved['native_group_observations'], source_summary=saved['source_summary'])
    from scar.optimization.constructive_planner import construct_implementations
    from scar.optimization.joint_planner import verify_execution_plan
    problems = [('strict_plan', strict), ('controlled_plan', controlled)]
    if documents['native_task_problem'] is not None:
        from scar.ir.implementation_plan import problem_from_dict
        problems.append(('native_task_plan', problem_from_dict(documents['native_task_problem'])))
    for name, problem in problems:
        plan = read_artifact(saved['files'][name])
        if plan['selected']:
            verification = verify_execution_plan(problem, plan)
            if name == 'controlled_plan':
                documents['schedule_verification'] = dict(schema='scar.controlled-schedule-verification',schema_version=1,**verification)
            elif name == 'native_task_plan':
                documents['native_task_schedule_verification'] = dict(
                    schema='scar.controlled-native-task-schedule-verification', schema_version=1,
                    **verification)
        elif _without_elapsed(plan) != _without_elapsed(construct_implementations(problem, max_seconds=saved['max_seconds'])):
            raise ValueError('unresolved controlled construction differs from its original scoped problem')
        documents[name] = plan
    if (set(documents) != set(saved['files']) or any(
            (saved['files'][key] is not None if value is None else read_artifact(saved['files'][key]) != value)
            for key,value in documents.items())):
        raise ValueError('controlled source graph, resources, costs or plan differ from cold replay')
    if saved != _metadata(saved['kind'], saved['admission'], saved['entry_profiles'], saved['max_seconds'],
            saved['files'], strict, controlled, documents['strict_plan'], documents['controlled_plan'],
            saved['cost_use'], saved['cost_qualification'], saved['compiler_feedback'],
            saved['entry_calls'], saved['native_group_observations'], saved['source_summary']):
        raise ValueError('controlled preparation summary changed its original qualification')
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('PIPELINE_FUSION', 'STATEFUL_SPECIALIZATION'))
    parser.add_argument('--admission'); parser.add_argument('--entry-profile', action='append', default=[])
    parser.add_argument('--cost-use', choices=('EXPLICIT_DECLARED_ANALYSIS', 'QUALIFIED_TARGET_TIMING'),
        default='EXPLICIT_DECLARED_ANALYSIS')
    parser.add_argument('--cost-qualification', help='retained entry-cost qualification for the requested timing use')
    parser.add_argument('--compiler-feedback', help='source-bound GCC optimization collection result')
    parser.add_argument('--entry-calls', help='bounded conditional entry CALL projection')
    parser.add_argument('--source-summary', help='cold-replayable source function dependency summary')
    parser.add_argument('--native-group-observations', help='frozen canonical native group observations; not a whole-call profile')
    parser.add_argument('--out'); parser.add_argument('--check'); parser.add_argument('--max-seconds',type=float,default=30)
    args = parser.parse_args(argv)
    if args.check:
        saved = check(args.check)
        result = dict(source_graph_and_contract_replayed=True, strict_status=saved['strict_status'],
            controlled_status=saved['controlled_status'], native_loaded=False, deployment_authorized=False)
    elif not args.kind or not args.admission or not args.out:
        parser.error('--kind, --admission and --out required')
    else:
        saved = prepare(args.kind, args.admission, args.out, entry_profiles=args.entry_profile,max_seconds=args.max_seconds,
            cost_use=args.cost_use, cost_qualification=args.cost_qualification, compiler_feedback=args.compiler_feedback,
            entry_calls=args.entry_calls, native_group_observations=args.native_group_observations,
            source_summary=args.source_summary)
        result = dict(preparation=_reference(Path(args.out)/'preparation.json'),
            strict_status=saved['strict_status'], controlled_status=saved['controlled_status'],
            native_loaded=False, deployment_authorized=False)
    result['native_task_problem'] = saved['files'].get('native_task_problem')
    result['native_task_plan'] = saved['files'].get('native_task_plan')
    print(json.dumps(result,sort_keys=True)); return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Static demand for the actual, source-qualified semantic successor.

The enclosing controller admits the original source and revision.  This
adapter reads that revision's graphs/body/analyses, replays their structural
binding, and reuses the existing array-demand and shape authorities.  Static
slots are neither allocations nor evidence of physical data movement.
"""
from copy import deepcopy

from scar.compiler.model import plan_from_dict
from scar.compiler.pipeline_frontend import pipeline_digest
from scar.ir.execution_recipe import read_artifact
from scar.ir.v2.codec import semantic_from_dict
from scar.ir.v2.ids import OperationDefinitionID
from scar.ir.v2.semantic import OperationDefinition, OperationKind
from scar.optimization.demand import analyze_region_demand
from scar.optimization.pipeline_demand import (
    derive_pipeline_demand, _native_features, _quantity, _shape, _sum, _units,
)
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.semantic_rewrite import (
    analyze_static_region, require_current_analysis, rewrite_closed_region,
)


SCHEMA = 'scar.semantic-region-demand'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _unknown(reason, unit):
    return dict(status='UNKNOWN', value=None, unit=unit, reason=reason)


def _semantic_ref(document, pointer='#'):
    return dict(schema=document['schema'], semantic_sha256=semantic_sha256(document),
        locator=pointer)


def _analysis_content(analysis):
    # The producer's slot insertion order need not survive SG serialization.
    # Use/def rows are a keyed set; the separately recorded operation order
    # remains semantic and is never sorted here.
    result = deepcopy(analysis)
    result['data_use_def'] = sorted(result['data_use_def'], key=lambda row:row['slot']['wire'])
    return result


def _read_revision(revision, pipeline, selected_region):
    from scar.compiler.semantic_revision import SCHEMA as REVISION_SCHEMA, _selection, _body

    _need(type(revision) is dict and
        (revision.get('schema'), revision.get('schema_version')) == (REVISION_SCHEMA, 1),
        'a parent-admitted semantic region revision receipt is required')
    selection, alternative, rewrite = _selection(selected_region)
    _need(revision['selection'] == selection.as_dict()
        and revision['delta'] == alternative.delta.as_dict()
        and revision['selected_region_sha256'] == semantic_sha256(selected_region)
        and revision['decision'] == ('REWRITE' if rewrite else 'KEEP')
        and pipeline_digest(pipeline) == selected_region['qualification']['pipeline_sha256'],
        'revision selection or source pipeline differs from the admitted OIR')
    root = OperationDefinitionID('source-region:' + selection.region.value)
    members = [row['projected_id'] for row in selected_region['projection']['operations']]
    _need(revision['root'] == root.as_dict()
        and revision['projection']['root'] == root.as_dict()
        and revision['projection']['members'] == members,
        'revision root or original operation projection differs')
    files = revision['files']
    before_wire, after_wire = read_artifact(files['before']), read_artifact(files['after'])
    before, after = semantic_from_dict(before_wire), semantic_from_dict(after_wire)
    analyses = {}
    for name, graph in (('before', before), ('after', after)):
        analysis = read_artifact(files[name + '_analysis'])
        require_current_analysis(graph, analysis, root=root)
        _need(_analysis_content(analysis) == _analysis_content(analyze_static_region(graph, root))
            and analysis['revision'] == revision[name + '_revision'],
            name + ' analysis differs from the actual graph revision/root')
        analyses[name] = analysis
    _need({node.id.as_dict()['wire'] for node in before.children_of(root)}
        == {row['wire'] for row in members}, 'before graph has a different source CALL body')
    body_wire = None
    if rewrite:
        _need(files['body'] is not None, 'a rewritten composite requires its actual body artifact')
        body_wire = read_artifact(files['body'])
        _need(body_wire == _body(pipeline, selected_region, alternative),
            'actual composite body differs from the selected source fusion certificate')
        parent = before.definitions[root]
        replacement = OperationDefinition(OperationDefinitionID(alternative.delta.added_operations[0]),
            OperationKind.REGION, 'Source-qualified fused region', code_id=parent.code_id,
            source_file=parent.source_file, parent_id=root,
            contract=parent.contract, effect_summary=parent.effect_summary)
        expected, mapping = rewrite_closed_region(before, root, replacement, alternative,
            original_graph=files['before'], body_expansion=files['body'])
        _need(expected.to_dict() == after_wire and mapping == revision['mapping'],
            'after graph is not the actual selected source-region replacement')
        _need(len(after.children_of(root)) == 1
            and after.children_of(root)[0].id == replacement.id,
            'rewritten demand requires the actual single active composite')
    else:
        _need(files['body'] is None and before_wire == after_wire,
            'KEEP must retain the original graph without a fused body')
    return before, after, root, analyses['after'], body_wire, rewrite


def _features(demand, elements, *, source):
    """Instantiate existing coefficients; do not classify expressions here."""
    if source:
        baseline = demand['baseline']
        retained = baseline['reference'] == 'RETAINED_OPTIMIZER_SOURCE_DAG'
        from scar.optimization.pipeline_demand import _original_features
        features = _original_features(demand, elements)
        features.update(region_body_evaluations=1, output_elements=elements,
            logical_array_reads=_quantity(baseline['logical_array_reads'], elements) if retained else None,
            logical_array_writes=_quantity(baseline['logical_array_writes'], elements) if retained else None)
    else:
        features = _native_features(demand, elements)
        features['region_body_evaluations'] = features.pop('native_region_calls')
        features.update(cumulative_allocation_bytes=features['output_bytes'],
            logical_array_reads=_quantity(demand['total_work']['array_loads'], elements),
            logical_array_writes=_quantity(demand['total_work']['output_stores'], elements))
    return features


def _feature_units(features):
    extra = dict(region_body_evaluations='static_body_evaluations/caller_activation',
        logical_array_reads='logical_float64_loads/caller_activation',
        logical_array_writes='logical_float64_stores/caller_activation')
    return {**_units({key:value for key, value in features.items() if key not in extra}),
        **{key:extra[key] for key in features if key in extra}}


def _demand_record(features, refs, peak):
    return dict(features=features, feature_units=_feature_units(features),
        feature_sources={key:deepcopy(refs) for key in features},
        peak_payload=peak, source_refs=deepcopy(refs))


def derive_semantic_region_demand(revision, pipeline, selected_region, blueprint):
    """Consume a cold-checked receipt; never replay source admission or execute it.

    REWRITE demand describes the raw body named by the successor composite.
    The separately verified optimized blueprint is a prospective lowering,
    with its own identity and demand, not a substitute for that raw body.
    """
    before, after, root, analysis, body_wire, rewrite = _read_revision(revision, pipeline, selected_region)
    parent = derive_pipeline_demand(pipeline, blueprint)
    _need(parent['pipeline_sha256'] == selected_region['qualification']['pipeline_sha256'],
        'blueprint demand belongs to a different source pipeline')
    files = revision['files']
    parent_ref = _semantic_ref(parent)
    root_record = after.definitions[root]
    q = selected_region['qualification']
    _need(root_record.metadata['Q'] == q['Q']
        and root_record.metadata['world_routes'] == selected_region['projection']['world_routes'],
        'successor boundary Q or exact original world routes differ')
    active, source_records = [], []
    before_indices = {row['id']['wire']:index for index, row in enumerate(before.to_dict()['definitions'])}
    after_indices = {row['id']['wire']:index for index, row in enumerate(after.to_dict()['definitions'])}
    for index, call in enumerate(pipeline.calls):
        leaf = parent['leaves'][index]
        demand = analyze_region_demand(call.plan)
        features = _features(demand, leaf['output_elements'], source=True)
        wire = selected_region['projection']['operations'][index]['projected_id']['wire']
        refs = [dict(artifact=files['before'], locator='#/definitions/' + str(before_indices[wire])),
            {**parent_ref, 'locator':'#/leaves/' + str(index)}, *deepcopy(leaf['source_refs'])]
        peak = dict(actual=_unknown('Original Python/NumPy intermediate release times are not established.', 'bytes'),
            scope='MODELED_ARRAY_PAYLOAD_EXCLUDES_BORROWED_INPUTS_ALLOCATOR_METADATA_AND_RSS')
        source_records.append(dict(operation=selected_region['projection']['operations'][index]['projected_id'],
            call=call.id, **_demand_record(features, refs, peak)))
    source_features = {key:_sum([row['features'][key] for row in source_records])
        for key in source_records[0]['features']}
    source_peak = dict(actual=_unknown('Source graph last-use is not Python reference release or process RSS.', 'bytes'),
        cumulative_allocation_payload_upper_bound=dict(value=source_features['cumulative_allocation_bytes'],
            unit='logical_bytes', source_refs=[parent_ref],
            scope='UPPER_BOUND_FOR_MODELED_ALLOCATED_ARRAY_PAYLOAD_ONLY_NOT_MEASURED_PEAK'),
        scope='MODELED_ARRAY_PAYLOAD_EXCLUDES_BORROWED_INPUTS_ALLOCATOR_METADATA_AND_RSS')
    source = _demand_record(source_features, [parent_ref], source_peak)
    source['operations'] = source_records
    by_wire = {node.id.wire:node for node in after.children_of(root)}
    if rewrite:
        plan = plan_from_dict(body_wire)
        shape, elements = _shape(plan, blueprint['original_memory_plan']['input_shapes'], parent['output_shape'])
        demand = analyze_region_demand(plan)
        features = _features(demand, elements, source=False)
        refs = [dict(artifact=files['after'], locator='#/definitions/' +
            str(after_indices[analysis['operation_order'][0]])),
            dict(artifact=files['body'], locator='#/region'),
            dict(artifact=files['body'], locator='#/proof/fusion_certificate')]
        peak = dict(actual=dict(value=features['output_bytes'], unit='logical_bytes',
            source_refs=deepcopy(refs)),
            scope='RAW_FUSED_MODEL_FRESH_FINAL_ARRAY_ONLY_EXCLUDES_BORROWED_INPUTS_BOUNDARY_WORK_AND_RSS')
        record = _demand_record(features, refs, peak)
        record.update(body_plan_sha256=semantic_sha256(body_wire), output_shape=shape,
            region_demand_sha256=semantic_sha256(demand))
        operation = by_wire[analysis['operation_order'][0]]
        active.append(dict(operation=operation.id.as_dict(), kind=operation.kind.value,
            representation='RAW_SELECTED_COMPOSITE_BODY', **record))
    else:
        original = {row['operation']['wire']:row for row in source_records}
        for wire in analysis['operation_order']:
            operation = by_wire[wire]
            row = deepcopy(original[wire])
            row.update(kind=operation.kind.value, representation='ORIGINAL_EAGER_SOURCE_CALL')
            active.append(row)
        features = deepcopy(source_features)
        peak = deepcopy(source_peak)
    active_demand = _demand_record(features, [dict(artifact=files['after'], locator='#')], peak)
    active_demand.update(operations=active, static_operation_count=len(active),
        graph_revision=analysis['revision'], root=root.as_dict())
    # This is a different body with a complete plan identity. Its eventual
    # resource, cost and delivery correspondence remains the controller's job.
    lowering_plan = plan_from_dict(blueprint['optimized_plan'])
    _, lowering_elements = _shape(lowering_plan, blueprint['original_memory_plan']['input_shapes'], parent['output_shape'])
    lowering = _features(analyze_region_demand(lowering_plan), lowering_elements, source=False)
    lowering_refs = [_semantic_ref(blueprint, '#/optimized_plan')]
    lowering_demand = _demand_record(lowering, lowering_refs,
        dict(actual=dict(value=lowering['output_bytes'], unit='logical_bytes', source_refs=lowering_refs),
            scope='LOWERING_FRESH_FINAL_ARRAY_ONLY_EXCLUDES_BORROWED_INPUTS_BOUNDARY_WORK_AND_RSS'))
    lowering_demand.update(plan_sha256=semantic_sha256(blueprint['optimized_plan']),
        strategy_sha256=blueprint['strategy_sha256'], active_semantic_body=False,
        active_lowering_selected=False, raw_body_to_lowering_binding_required=True)
    physical = deepcopy(parent['physical'])
    for field, reason, unit in (
        ('whole_entry_cpu_execution_domain', 'Neither source slots nor requested workers prove whole-entry CPU binding or capacity.', 'hardware_thread_units'),
        ('opaque_runtime_memory', 'Logical array payload excludes allocator overhead, thread scratch and process RSS.', 'bytes'),
        ('physical_memory_residency', 'Static source routes do not establish RAM/NUMA/cache residency.', 'memory_domain')):
        physical[field] = _unknown(reason, unit)
    frontier = deepcopy(parent['frontier'])
    for key, value in physical.items():
        frontier.append(dict(field='physical.' + key, reason=value['reason']))
    return dict(schema=SCHEMA, schema_version=1,
        graph_revision=analysis['revision'], root=root.as_dict(), decision=revision['decision'],
        selection=deepcopy(revision['selection']), selected_region_sha256=semantic_sha256(selected_region),
        semantic_revision_sha256=semantic_sha256(revision), pipeline_sha256=parent['pipeline_sha256'],
        blueprint_sha256=parent['blueprint_sha256'], source_baseline=source, active=active_demand,
        implementation_lowering_demand=lowering_demand,
        boundary=dict(input_slots=[key.as_dict() for key in root_record.input_slots],
            output_slots=[key.as_dict() for key in root_record.output_slots],
            world_slots=[key.as_dict() for key in root_record.state_slots],
            source_ports=deepcopy(selected_region['projection']['boundary_ports']),
            world_routes=deepcopy(selected_region['projection']['world_routes']),
            contract=deepcopy(after.contracts[root_record.contract].as_dict()),
            effect_summary=deepcopy(after.effects[root_record.effect_summary].as_dict()),
            required_runtime_guards=deepcopy(q['required_runtime_guards']), Q=deepcopy(q['Q'])),
        input_shapes=deepcopy(parent['input_shapes']), output_shape=deepcopy(parent['output_shape']),
        shape_qualification='STATIC_INSTANTIATION_CONDITIONAL_ON_UNEXECUTED_RUNTIME_GUARDS_NOT_RUNTIME_INVARIANCE',
        runtime_boundary=deepcopy(parent['runtime_boundary']), physical=physical,
        required_hardware_cost_fields=deepcopy(parent['required_hardware_cost_fields']), frontier=frontier,
        feature_definitions=dict(arithmetic='Logical source/scalar-DAG operations, including scalar setup once per invocation; not emitted instructions.',
            logical_array_reads='Existing demand analyzer float64 load count under the guarded shape model; excludes guard/finite scans and does not predict DRAM transfers.',
            logical_array_writes='Existing demand analyzer float64 store count under the guarded shape model; not physical write traffic.',
            array_allocations='Cumulative logical fresh arrays, including each result; not allocator calls, memory-pool slots or peak RSS.',
            output_elements='Sum of modeled body output elements for source/KEEP; final output elements for the single raw composite/lowering.',
            peak_payload='Explicitly scoped logical payload only; original Python actual peak remains UNKNOWN.'),
        source_refs=[dict(artifact=files[key], locator='#') for key in ('before', 'after', 'after_analysis')]
            + ([] if files['body'] is None else [dict(artifact=files['body'], locator='#')])
            + deepcopy(selected_region['source_refs']) + [parent_ref],
        scope='ACTUAL_SEMANTIC_SUCCESSOR_STATIC_LOGICAL_ARRAY_DEMAND_NOT_PHYSICAL_TRAFFIC_OR_TIME',
        source_admission='PARENT_COLD_CHECK_REQUIRED_NOT_PERFORMED_BY_THIS_ADAPTER',
        runtime_guards_executed=False, source_bodies_copied=False,
        runtime_values_created=False, storage_lifetimes_inferred=False,
        target_executed=False, native_compiled=False, native_loaded=False,
        timing_prediction_available=False, physical_memory_traffic_predicted=False,
        execution_authorized=False)


def verify_semantic_region_demand(revision, pipeline, selected_region, blueprint, report):
    _need(type(report) is dict and (report.get('schema'), report.get('schema_version')) == (SCHEMA, 1),
        'versioned semantic successor demand required')
    expected = derive_semantic_region_demand(revision, pipeline, selected_region, blueprint)
    _need(semantic_sha256(expected) == semantic_sha256(report),
        'semantic demand differs from the actual successor/body/source/shape replay')
    return expected


__all__ = ['SCHEMA', 'derive_semantic_region_demand', 'verify_semantic_region_demand']

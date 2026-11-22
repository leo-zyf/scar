"""Static demands for original pipeline calls, fused bodies and semantic successors.

The enclosing controller admits the original source and revision.  This
adapter reads that revision's graphs/body/analyses, replays their structural
binding, and reuses the existing array-demand and shape authorities.  Static
slots are neither allocations nor evidence of physical data movement.
"""
from copy import deepcopy

from scar.compiler.model import document as plan_document, plan_from_dict
from scar.compiler.pipeline_frontend import document as pipeline_document, pipeline_digest, validate_pipeline
from scar.ir.execution_recipe import read_artifact
from scar.ir.v2.codec import semantic_from_dict
from scar.ir.v2.ids import OperationDefinitionID
from scar.ir.v2.semantic import OperationDefinition, OperationKind
from scar.optimization.demand import analyze_region_demand
from scar.optimization.resource_solver import _shape_instantiation, semantic_sha256
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


PIPELINE_DEMAND_SCHEMA = 'scar.pipeline-demand'
SUCCESSOR_SCHEMA = 'scar.pipeline-successor-cost-demand'
_OPS = ('add', 'sub', 'mul', 'div', 'neg')
_MISSING_SOURCE_WORK = ('The region demand has no qualified original source arithmetic classification; '
    'optimized/CSE arithmetic is not substituted for original eager work.')


def _total(coefficient, elements):
    if elements == 0:
        return 0
    return None if coefficient is None or elements is None else coefficient * elements


def _sum(values):
    return None if any(value is None for value in values) else sum(values)


def _quantity(quantity, elements):
    return _total(quantity['per_output_element'], elements)


def _arithmetic(array_counts, scalar_counts, elements):
    result = {}
    for operation in _OPS:
        array = array_counts.get(operation)
        scalar = scalar_counts.get(operation)
        _need(all(value is None or type(value) is int and value >= 0 for value in (array, scalar)),
            'arithmetic coefficients must be exact nonnegative counts or UNKNOWN')
        result['arithmetic.' + operation] = _sum([_total(array, elements), scalar])
    return result


def _original_features(demand, elements):
    baseline = demand['baseline']
    retained = baseline['reference'] == 'RETAINED_OPTIMIZER_SOURCE_DAG'
    result = dict(array_allocations=baseline['array_allocations_per_call'] if retained else None,
        cumulative_allocation_bytes=_quantity(baseline['cumulative_array_allocation_bytes'], elements)
            if retained else None,
        logical_payload_bytes=_quantity(baseline['logical_payload_transfer_bytes'], elements)
            if retained else None)
    # Only the shared demand analyzer may classify original work.  A future
    # supported classification can fill these fields without another pipeline
    # implementation of scalar/type/CSE accounting.
    array = baseline.get('arithmetic_per_output_element', {}) if retained else {}
    scalar = baseline.get('scalar_arithmetic_per_call', {}) if retained else {}
    result.update(_arithmetic(array, scalar, elements))
    return result


def _native_features(demand, elements):
    result = dict(native_region_calls=demand['per_call_setup']['region_invocations'],
        output_elements=elements, array_allocations=demand['per_call_setup']['logical_output_allocations'],
        output_bytes=_quantity(demand['fused']['fresh_output_bytes'], elements),
        logical_payload_bytes=_quantity(demand['fused']['logical_payload_transfer_bytes'], elements),
        input_bindings=demand['per_call_setup']['input_bindings'])
    result.update(_arithmetic(demand['per_output_element']['arithmetic'],
        demand['per_call_setup']['scalar_arithmetic'], elements))
    return result


def _unknown_features(features, prefix, frontier, *, source=False):
    for key, value in features.items():
        _need(value is None or type(value) is int and value >= 0,
            'demand features must be exact nonnegative integers or UNKNOWN')
        if value is None:
            frontier.append(dict(field=prefix + '.' + key, reason=(
                _MISSING_SOURCE_WORK if key.startswith('arithmetic.') and source else
                'Original retained eager baseline or shape instantiation is incomplete.')))


def _shape(plan, parameters, expected):
    _, shape, issues = _shape_instantiation(plan, parameters)
    _need(not issues and shape is not None and list(shape) == expected,
        'demand shapes differ from the replayed original value/leaf input routing')
    return list(shape), shape[0] * shape[1]


def _reference(blueprint_sha, pointer):
    return dict(schema='scar.pipeline-fusion-blueprint', semantic_sha256=blueprint_sha,
        locator=pointer)


def _symbolic(demand):
    """Demand coefficients only; keep source graphs/maps in their parents."""
    return dict(region_demand_sha256=semantic_sha256(demand),
        per_output_element=deepcopy(demand['per_output_element']),
        per_call_setup=deepcopy(demand['per_call_setup']),
        baseline=deepcopy(demand['baseline']),
        fused=deepcopy(demand['fused']),
        source_graphs_copied=False)


def _units(features):
    result = {}
    for key in features:
        if key.startswith('arithmetic.'):
            result[key] = 'scalar_DAG_operations/caller_activation'
        elif key.endswith('_bytes'):
            result[key] = 'logical_bytes/caller_activation'
        elif key.endswith('_calls'):
            result[key] = 'invocations/caller_activation'
        elif key.endswith('_allocations'):
            result[key] = 'logical_array_allocations/caller_activation'
        elif key.endswith('_bindings'):
            result[key] = 'logical_input_bindings/caller_activation'
        elif key == 'output_elements':
            result[key] = 'float64_elements/caller_activation'
        else:
            raise ValueError('undefined pipeline demand feature unit: ' + key)
    return result


def derive_pipeline_demand(pipeline, blueprint):
    """Replay the existing blueprint, then instantiate and sum region demands.

    Blueprint replay checks the original strategy and backend receipts without
    compilation, CDLL or workload execution.  It does not perform the original
    F4 source replay or discharge any of its live runtime guards.
    """
    from scar.optimization.macro_pass import replay_fusion_blueprint

    pipeline = validate_pipeline(pipeline)
    return _derive_qualified_pipeline_demand(pipeline, replay_fusion_blueprint(pipeline, blueprint))


def _derive_qualified_pipeline_demand(pipeline, value):
    blueprint_sha = semantic_sha256(value)
    memory = value['original_memory_plan']
    wire = pipeline_document(pipeline)
    values = {row['id']:row for row in memory['values']}
    frontier, leaves, cache = [], [], {}
    required_hardware = {}
    for index, call in enumerate(pipeline.calls):
        key = wire['calls'][index]['plan_id']
        plan_sha = semantic_sha256(plan_document(call.plan))
        if plan_sha not in cache:
            cache[plan_sha] = analyze_region_demand(call.plan)
        demand = cache[plan_sha]
        parameters = {name:values[logical]['shape'] for name, logical in zip(
            call.plan.region.parameters, call.inputs, strict=True)}
        shape, elements = _shape(call.plan, parameters, values[call.output]['shape'])
        original = _original_features(demand, elements)
        native = _native_features(demand, elements)
        _unknown_features(original, 'leaves.' + str(index) + '.original_features', frontier, source=True)
        _unknown_features(native, 'leaves.' + str(index) + '.native_features', frontier)
        required_hardware.update(deepcopy(demand['required_hardware_cost_fields']))
        leaves.append(dict(call=call.id, index=index, leaf_plan_id=key, leaf_plan_sha256=plan_sha,
            input_mapping=[dict(parameter=name, value=logical, shape=deepcopy(values[logical]['shape']),
                source_refs=deepcopy(values[logical]['source_refs'])) for name, logical in zip(
                    call.plan.region.parameters, call.inputs, strict=True)],
            output_value=call.output, output_shape=shape, output_elements=elements,
            original_features=original, native_features=native,
            original_feature_units=_units(original), native_feature_units=_units(native),
            symbolic_demand=_symbolic(demand),
            source_refs=[_reference(blueprint_sha, '#/pipeline/calls/' + str(index)),
                _reference(blueprint_sha, '#/pipeline/leaf_plans/' + key),
                _reference(blueprint_sha, '#/pipeline/leaf_plans/' + key + '/region/source'),
                _reference(blueprint_sha, '#/pipeline/leaf_plans/' + key + '/proof/source_operation_map')]))
    plan = plan_from_dict(value['optimized_plan'])
    fused_demand = analyze_region_demand(plan)
    final = values[pipeline.output]
    shape, elements = _shape(plan, memory['input_shapes'], final['shape'])
    native = _native_features(fused_demand, elements)
    fused = dict(native_region_calls=native['native_region_calls'], output_elements=elements,
        fused_array_allocations=native['array_allocations'], fused_output_bytes=native['output_bytes'],
        fused_logical_payload_bytes=native['logical_payload_bytes'],
        fused_input_bindings=native['input_bindings'],
        **{key:native[key] for key in native if key.startswith('arithmetic.')})
    original = dict(python_region_calls=1, python_leaf_calls=len(leaves), output_elements=elements,
        **{key:_sum([row['original_features'][key] for row in leaves])
            for key in leaves[0]['original_features']})
    leaf_native = {key:_sum([row['native_features'][key] for row in leaves])
        for key in leaves[0]['native_features']}
    # A leaf sum's output_elements counts all intermediate/final CALL outputs,
    # unlike the public region output_elements in original/fused features.
    _unknown_features(original, 'original.features', frontier, source=True)
    _unknown_features(fused, 'fused.features', frontier)
    _unknown_features(leaf_native, 'leaf_native.features', frontier)
    required_hardware.update(deepcopy(fused_demand['required_hardware_cost_fields']))
    boundary = {name:dict(status='UNKNOWN', value=None, unit='seconds/caller_activation', reason=reason)
        for name, reason in (
            ('original_python_boundary', 'Python caller/callee dispatch and NumPy service costs are not durations derived from DAG counts.'),
            ('protected_caller_guards', 'Complete original source/callee/input/shape/alias guards and fallback costs must be costed explicitly.'),
            ('finite_input_and_fp_status', 'Finite-input checks and floating-environment/status work are separate runtime boundary costs; logical load counts are not scan counts.'),
            ('native_entry_and_descriptors', 'Python/ABI argument and output descriptor construction and native dispatch costs are not zero.'),
            ('allocation_service', 'Logical fresh-array counts and payload bytes do not establish allocator latency.'),
        )}
    for name, missing in boundary.items():
        frontier.append(dict(field='runtime_boundary.' + name, reason=missing['reason']))
    physical = deepcopy(fused_demand['unknown'])
    for name, missing in physical.items():
        frontier.append(dict(field='physical.' + name, reason=missing['reason']))
    for name, missing in required_hardware.items():
        frontier.append(dict(field='required_hardware_cost_fields.' + name, reason=missing['reason']))
    return dict(schema=PIPELINE_DEMAND_SCHEMA, schema_version=1, pipeline_sha256=pipeline_digest(pipeline),
        blueprint_sha256=blueprint_sha, strategy_sha256=value['strategy_sha256'],
        original_memory_plan_sha256=value['original_memory_plan_sha256'],
        optimized_plan_sha256=value['optimized_plan_sha256'],
        input_shapes=deepcopy(memory['input_shapes']), output_shape=shape,
        source_refs=deepcopy(value['source_refs']) + [_reference(blueprint_sha, '#/strategy/fusion_certificate')],
        original=dict(features=original, feature_units=_units(original),
            scope='One original Python caller activation; eager baseline summed separately for every source CALL, including repeated shared leaves.'),
        leaf_native=dict(features=leaf_native, feature_units=_units(leaf_native),
            scope='Sum of individually lowered leaf invocations; logical allocations, not fixed-pool host allocation events or host runtime costs.'),
        leaves=leaves,
        fused=dict(features=fused, feature_units=_units(fused), symbolic_demand=_symbolic(fused_demand),
            source_refs=[_reference(blueprint_sha, '#/optimized_plan'),
                _reference(blueprint_sha, '#/optimized_plan/proof/source_operation_map')],
            scope='One final fused native invocation; array work multiplied by final output elements and scalar setup counted once.'),
        feature_definitions=dict(array_allocations='Cumulative logical fresh arrays, including each final result; zero-payload arrays still count as allocations.',
            cumulative_allocation_bytes='Sum of original eager array payload allocation bytes across all CALLs; includes intermediate and final results, excludes existing inputs and allocator metadata; not peak live memory.',
            logical_payload_bytes='Logical scalar-DAG array read plus write bytes from optimizer._cost, including original eager intermediate materializations; not DRAM traffic.',
            arithmetic='Per-output-element array arithmetic times each CALL output size plus original/native scalar setup once per invocation; zero is proven absence, None is UNKNOWN.',
            fused_input_bindings='Logical formal descriptors retained by the fused signature, including inputs whose contents are unused; not data loads.',
            output_elements='Public final elements for original/fused; sum of every CALL output size for leaf_native.'),
        runtime_boundary=boundary, physical=physical, required_hardware_cost_fields=required_hardware,
        frontier=frontier, scope='GUARDED_STATIC_SYMBOLIC_DEMAND_NOT_MACHINE_INSTRUCTIONS_PHYSICAL_TRAFFIC_OR_TIME',
        accounting_authority='scar.optimization.demand.analyze_region_demand / optimizer._cost / resource_solver._shape_instantiation',
        source_replayed_by_parent_required=True, source_replay_performed=False,
        runtime_guards_executed=False, finite_input_checks_executed=False,
        native_compiled=False, native_loaded=False, target_executed=False,
        timing_prediction_available=False, physical_memory_traffic_predicted=False)


def _check_pipeline_demand_schema(report):
    _need(type(report) is dict and (report.get('schema'), report.get('schema_version')) == (PIPELINE_DEMAND_SCHEMA, 1),
        'versioned pipeline demand report required')


def _compare_pipeline_demand(report, expected):
    _need(semantic_sha256(expected) == semantic_sha256(report),
        'pipeline demand differs from original CALL/shape/baseline/fusion replay')
    return expected


def _verify_admitted_pipeline_demand(admitted, report):
    _check_pipeline_demand_schema(report)
    return _compare_pipeline_demand(report,
        _derive_qualified_pipeline_demand(admitted.pipeline, admitted.blueprint))


def verify_pipeline_demand(pipeline, blueprint, report):
    """Replay every parent/count/shape and compare the complete demand document."""
    _check_pipeline_demand_schema(report)
    return _compare_pipeline_demand(report, derive_pipeline_demand(pipeline, blueprint))


def derive_successor_pipeline_cost_demand(region, blueprint, preparation_ref):
    """Project the published candidate demand into the protected-entry cost basis.

    The controller cold-checks the parent once. This projection verifies the
    bound artifact identities on each replay and does not analyze the CALLs or
    re-derive a candidate graph. Keep the raw composite and lowering distinct.
    """
    from scar.compiler.semantic_revision import SCHEMA as REVISION_SCHEMA

    parent = read_artifact(preparation_ref)
    _need((parent['schema'], parent['schema_version']) == ('scar.macro-pipeline-preparation', 2),
        'versioned macro preparation required for successor costing')
    files = parent['files']
    _need(region == read_artifact(files['region']) and blueprint == read_artifact(files['blueprint']),
        'protected-entry region or blueprint differs from the published macro preparation')
    revision_ref, demand_ref = files['candidate_revision'], files['candidate_demand']
    _need(revision_ref is not None and demand_ref is not None,
        'macro preparation has no qualified published fused successor to cost')
    revision, semantic = read_artifact(revision_ref), read_artifact(demand_ref)
    _need((revision['schema'], revision['schema_version'], revision['decision']) == (REVISION_SCHEMA, 1, 'REWRITE')
        and (semantic['schema'], semantic['schema_version'], semantic['decision']) == (SCHEMA, 1, 'REWRITE'),
        'versioned fused graph and its semantic demand required')
    lowered = semantic['implementation_lowering_demand']
    active = semantic['active']
    qualification = region['qualification']
    bounds = semantic['boundary']
    _need(semantic['semantic_revision_sha256'] == semantic_sha256(revision)
        and semantic['selected_region_sha256'] == revision['selected_region_sha256']
        and semantic['graph_revision'] == active['graph_revision'] == revision['after_revision']
        and semantic['root'] == active['root'] == revision['root']
        and semantic['pipeline_sha256'] == blueprint['pipeline_sha256'] == qualification['pipeline_sha256']
        and semantic['blueprint_sha256'] == semantic_sha256(blueprint)
        and lowered['plan_sha256'] == blueprint['optimized_plan_sha256'] == semantic_sha256(blueprint['optimized_plan'])
        and lowered['strategy_sha256'] == blueprint['strategy_sha256']
        and semantic['input_shapes'] == qualification['input_shapes']
        and semantic['output_shape'] == qualification['output_shape']
        and bounds['Q'] == qualification['Q']
        and bounds['required_runtime_guards'] == qualification['required_runtime_guards'],
        'successor graph, implementation lowering, or guarded source contract differs')
    raw_body = revision['files']['body']
    raw_body_sha = semantic_sha256(read_artifact(raw_body))
    _need(active['static_operation_count'] == len(active['operations']) == 1
        and raw_body_sha == active['operations'][0]['body_plan_sha256'],
        'successor demand must name the actual raw composite body')
    binding = dict(candidate_revision=revision_ref, candidate_demand=demand_ref,
        original_graph=revision['files']['before'], original_graph_revision=revision['before_revision'],
        candidate_graph=revision['files']['after'], graph_revision=revision['after_revision'],
        semantic_revision_sha256=semantic['semantic_revision_sha256'],
        blueprint_sha256=semantic['blueprint_sha256'], pipeline_sha256=semantic['pipeline_sha256'],
        raw_body=raw_body, raw_body_plan_sha256=raw_body_sha,
        optimized_plan_sha256=lowered['plan_sha256'], strategy_sha256=lowered['strategy_sha256'])
    source, native = semantic['source_baseline']['features'], lowered['features']
    _need(source['region_body_evaluations'] == len(semantic['source_baseline']['operations'])
        and native['region_body_evaluations'] == 1,
        'protected entry requires the original CALL count and exactly one fused invocation')
    original = dict(python_region_calls=1, python_leaf_calls=source['region_body_evaluations'],
        output_elements=native['output_elements'],
        **{name:source[name] for name in ('array_allocations', 'cumulative_allocation_bytes', 'logical_payload_bytes')},
        **{name:value for name, value in source.items() if name.startswith('arithmetic.')})
    fused = dict(native_region_calls=native['region_body_evaluations'],
        output_elements=native['output_elements'], fused_array_allocations=native['array_allocations'],
        fused_output_bytes=native['output_bytes'], fused_logical_payload_bytes=native['logical_payload_bytes'],
        fused_input_bindings=native['input_bindings'],
        **{name:value for name, value in native.items() if name.startswith('arithmetic.')})
    frontier = deepcopy(semantic['frontier'])
    _unknown_features(original, 'original.features', frontier, source=True)
    _unknown_features(fused, 'fused.features', frontier)
    return dict(schema=SUCCESSOR_SCHEMA, schema_version=1, macro_preparation=preparation_ref,
        candidate_binding=binding, Q=deepcopy(bounds['Q']),
        original=dict(features=original, feature_units=_units(original)),
        fused=dict(features=fused, feature_units=_units(fused)),
        source_refs=[dict(artifact=demand_ref, locator='#' + path) for path in (
            '/source_baseline', '/active', '/implementation_lowering_demand', '/boundary/Q')],
        accounting_authority='PUBLISHED_SOURCE_BASELINE_AND_IMPLEMENTATION_LOWERING_DEMAND',
        runtime_boundary=deepcopy(semantic['runtime_boundary']), physical=deepcopy(semantic['physical']),
        required_hardware_cost_fields=deepcopy(semantic['required_hardware_cost_fields']), frontier=frontier)


def verify_successor_pipeline_cost_demand(region, blueprint, report):
    _need(type(report) is dict and (report.get('schema'), report.get('schema_version')) == (SUCCESSOR_SCHEMA, 1),
        'versioned successor protected-entry demand required')
    expected = derive_successor_pipeline_cost_demand(region, blueprint, report['macro_preparation'])
    _need(semantic_sha256(expected) == semantic_sha256(report),
        'protected-entry demand differs from the published semantic successor')
    return expected


__all__ = [
    'SCHEMA', 'PIPELINE_DEMAND_SCHEMA', 'SUCCESSOR_SCHEMA',
    'derive_semantic_region_demand', 'verify_semantic_region_demand',
    'derive_pipeline_demand', 'verify_pipeline_demand',
    'derive_successor_pipeline_cost_demand', 'verify_successor_pipeline_cost_demand',
]

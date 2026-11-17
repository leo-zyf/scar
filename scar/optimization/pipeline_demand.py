"""Compose existing region demands over a replayed full-fusion blueprint.

All totals are static logical work per caller activation under the retained
shape/array contract.  They are neither physical traffic nor complete boundary
costs.  The controller owns original F4 admission; this module never reopens a
source program, imports a workload, compiles native code or loads a library.
"""
from copy import deepcopy

from scar.compiler.model import document as plan_document, plan_from_dict
from scar.compiler.pipeline_model import document as pipeline_document, pipeline_digest, validate_pipeline
from scar.optimization.demand import analyze_region_demand
from scar.optimization.resource_solver import _shape_instantiation, semantic_sha256


SCHEMA = 'scar.pipeline-demand'
_OPS = ('add', 'sub', 'mul', 'div', 'neg')
_MISSING_SOURCE_WORK = ('The region demand has no qualified original source arithmetic classification; '
    'optimized/CSE arithmetic is not substituted for original eager work.')


def _need(condition, message):
    if not condition:
        raise ValueError(message)


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
    from scar.compiler.pipeline_fusion_delivery import verify_fusion_blueprint
    pipeline = validate_pipeline(pipeline)
    value = verify_fusion_blueprint(pipeline, blueprint['original_memory_plan'],
        blueprint['target'], blueprint, available_cpu_ids=blueprint['execution']['allowed_cpu_ids'])
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
    return dict(schema=SCHEMA, schema_version=1, pipeline_sha256=pipeline_digest(pipeline),
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


def verify_pipeline_demand(pipeline, blueprint, report):
    """Replay every parent/count/shape and compare the complete demand document."""
    _need(type(report) is dict and (report.get('schema'), report.get('schema_version')) == (SCHEMA, 1),
        'versioned pipeline demand report required')
    expected = derive_pipeline_demand(pipeline, blueprint)
    _need(semantic_sha256(expected) == semantic_sha256(report),
        'pipeline demand differs from original CALL/shape/baseline/fusion replay')
    return expected


__all__ = ['SCHEMA', 'derive_pipeline_demand', 'verify_pipeline_demand']

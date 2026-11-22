"""Resource-independent semantic boundary for one admitted temporal loop.

The original loop/body graphs remain shared. Interface labels here are local
recipe ports, never original ValueIDs, concrete content versions or storage IDs.
Resource admission is reused; this module neither chooses nor schedules CPUs.
"""
from copy import deepcopy
from pathlib import Path

from .loop_admission import admit_loop
from .loop_model import loop_digest, read_loop, validate_loop
from .model import document as plan_document, read_plan
from scar.compiler.pipeline_frontend import pipeline_digest
from .validation import validate_plan
from scar.optimization.loop_strategy import loop_execution_plan, verify_loop_strategy
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.loop-semantic-boundary'
_PREFIX = 'loop-semantic-boundary:'


def _reference(material, pointer):
    return dict(material=material, pointer=pointer)


def _source(source):
    return {field: getattr(source, field) for field in
            ('path', 'sha256', 'qualname', 'line', 'column', 'code_key')}


def _authority(pointer):
    return dict(kind='EXPLICIT_STRATEGY_SPECIALIZATION',
        observed_vs_declared='NOT_DISTINGUISHED_BY_INPUT_STRATEGY',
        source_constant_inferred=False, evidence_ref=_reference('strategy', pointer))


def boundary_digest(boundary):
    """Hash the semantic document, not a file receipt or an admission proof."""
    if (type(boundary) is not dict or boundary.get('schema') != SCHEMA
            or boundary.get('schema_version') != 1):
        raise ValueError('versioned loop semantic boundary required')
    value = dict(boundary)
    value.pop('id', None)
    return semantic_sha256(value)


def build_loop_boundary(loop, strategy, plan):
    """Replay loop/fusion/shape/P legality, then project its semantic boundary.

    A bound P goes through the shared complete CPU admission, including retained
    target receipts; no placement is chosen. An unbound P must exactly equal
    the original canonical strategy component. Resource counts, layout choices,
    budgets, costs and target receipts do not enter the returned boundary.
    """
    loop = read_loop(loop) if isinstance(loop, (str, Path)) else validate_loop(loop)
    actual = read_plan(plan) if isinstance(plan, (str, Path)) else plan
    validate_plan(actual)
    report = verify_loop_strategy(loop, strategy)
    if report['selected_strategy'] != 'TEMPORAL_FUSION' or report['frontier']:
        raise ValueError('closed exact temporal-fusion strategy required')
    canonical = loop_execution_plan(loop, report)
    if actual.resource_binding is not None:
        report, actual = admit_loop(loop, report, actual)
    else:
        validate_plan(actual)
        if plan_document(actual) != plan_document(canonical):
            raise ValueError('unbound body P differs from complete canonical semantic replay')
    if (actual.proof.get('supported') is not True or actual.proof.get('legal') is not True
            or actual.proof.get('frontiers') or actual.region.frontiers):
        raise ValueError('supported legal body P without unresolved frontiers required')

    shape = list(report['state_shape'])
    count = report['trip_count']
    certificate = report['body_fusion_certificate']
    source_map = deepcopy(canonical.proof['source_operation_map'])
    control_refs, backedge_refs = [], []
    for index, origin in enumerate(loop.origins):
        if origin.get('role') == 'single_shared_loop':
            control_refs.append(_reference('loop', '#/origins/' + str(index)))
            if 'backedge' in origin:
                backedge_refs.append(_reference('loop', '#/origins/' + str(index) + '/backedge'))
    body_path = list(loop.body_callee.lookup_path)
    guards = [dict(role='LOOP_BODY', environment_path=body_path,
        lookup_semantics='EACH_COMPONENT_USES_ITS_PARENT_FUNCTION_GLOBALS',
        source=_source(loop.body.source), source_ref=_reference('loop', '#/body/source'))]
    calls = []
    for index, (call, route) in enumerate(zip(loop.body.calls, certificate['calls'], strict=True)):
        if route['call_id'] != call.id:
            raise ValueError('fusion certificate changed original call ordering or identity')
        pointer = '#/body_fusion_certificate/calls/' + str(index)
        source_pointer = '#/body/leaf_plans/' + route['leaf_plan_id'] + '/region/source'
        guards.append(dict(role='BODY_LEAF', call_id=call.id,
            environment_path=body_path + list(call.callee.lookup_path),
            lookup_semantics='EACH_COMPONENT_USES_ITS_PARENT_FUNCTION_GLOBALS',
            source=_source(call.callee.source), source_ref=_reference('loop', source_pointer)))
        calls.append(dict(call_id=call.id, call_ref=_reference('loop', '#/body/calls/' + str(index)),
            leaf_plan_ref=_reference('loop', '#/body/leaf_plans/' + route['leaf_plan_id']),
            certificate_ref=_reference('strategy', pointer),
            source_expr_map_ref=_reference('strategy', pointer + '/source_expr_map'),
            expr_map_ref=_reference('strategy', pointer + '/expr_map')))

    value = dict(schema=SCHEMA, schema_version=1,
        profile='GUARDED_SINGLE_CARRY_ELEMENTWISE_LOOP',
        source=dict(loop=dict(id=loop.id, sha256=loop_digest(loop), source=_source(loop.source)),
            body=dict(id=loop.body.id, sha256=pipeline_digest(loop.body), source=_source(loop.body.source)),
            callee_guards=guards),
        parameters=list(loop.parameters),
        input=dict(id='state:entry', parameter=loop.state_parameter,
            parameter_index=loop.parameters.index(loop.state_parameter),
            body_parameter=loop.body.parameters[0], body_value=loop.body.input_values[0],
            shape=shape, dtype='float64', device='cpu', layout='C', ownership='CALLER_OWNED_READ_ONLY',
            shape_authority=_authority('#/input_shapes/' + loop.state_parameter),
            dtype_layout_authority='GUARDED_ORIGINAL_ARRAY_PROFILE_NOT_SOURCE_CONSTANTS'),
        output=dict(id='state:return', body_value=loop.body.output,
            expression=canonical.region.output, shape=list(report['output_shape']),
            dtype='float64', ownership='ONE_FRESH_CALLER_OWNED_FINAL', aliases_input=False,
            shape_authority='REPLAY_ORIGINAL_LEAVES_RAW_FUSION_AND_OPTIMIZED_DAG'),
        state=dict(id='state:carry', instance_parameters=['time'],
            identity_kind='BOUNDARY_INTERFACE_LABEL_NOT_ORIGINAL_VALUE_OR_STORAGE_ID',
            content_versions_inferred=False,
            body_input=dict(id='body:input', parameter=loop.body.parameters[0], value=loop.body.input_values[0]),
            body_output=dict(id='body:output', value=loop.body.output, expression=canonical.region.output),
            edge_scope='PER_COORDINATE_NO_GLOBAL_ITERATION_BARRIER_INFERRED',
            edges=[dict(source='state:entry', target='state:carry', condition='time == 0', kind='INITIAL'),
                   dict(source='state:carry', target='body:input', condition='0 <= time < count', kind='ARGUMENT'),
                   dict(source='body:output', target='state:carry', condition='next_time == time + 1', kind='BACKEDGE'),
                   dict(source='state:carry', target='state:return', condition='time == count', kind='NORMAL_RETURN')],
            source_control_refs=control_refs, source_backedge_refs=backedge_refs,
            source_port_evidence='RETAINED_F4_REFERENCES' if backedge_refs else 'TYPED_LOOP_CONTRACT_ONLY'),
        iteration=dict(axis='time', count_parameter=loop.trip_count_parameter,
            parameter_index=loop.parameters.index(loop.trip_count_parameter), count=count,
            count_type='EXACT_PYTHON_INT', count_authority=_authority('#/trip_count'),
            order='range(count)', start=0, step=1, unrolled=False, shared_body_instances=1,
            control_kind=report['control']['kind'],
            carry_policy=report['control']['array_carry'],
            nonpositive_count='OUTSIDE_THIS_SPECIALIZATION_ORIGINAL_INPUT_ALIAS_REQUIRES_FALLBACK'),
        parallelism=dict(spatial_axes=[dict(axis='row', extent=shape[0]), dict(axis='col', extent=shape[1])],
            partition_policy='DISJOINT_COORDINATES', ordered_axes=['time'],
            cross_coordinate_dependencies=False, numerical_reduction=False,
            temporal_carry='STRICT_SAME_COORDINATE_RECURRENCE',
            proof_scope='CLOSED_ELEMENTWISE_BODY_PROFILE_WITH_ORIGINAL_SHAPE_REPLAY'),
        dependencies=dict(
            input_ready='CALLER_STATE_ARGUMENT_READY_BEFORE_INITIAL_CARRY_LOAD',
            body_ready='ALL_ORIGINAL_SCALAR_DAG_OPERANDS_READY_BEFORE_THEIR_CONSUMER',
            carry_ready='BODY_OUTPUT_READY_BEFORE_NEXT_SAME_COORDINATE_ITERATION',
            output_ready='ALL_COORDINATES_COMPLETE_ALL_REQUIRED_ITERATIONS_BEFORE_NORMAL_RETURN_EXPORT',
            control='EXACT_COUNT_RANGE_CALLABLE_AND_ARRAY_GUARDS_HOLD_BEFORE_EXECUTION',
            failure='NO_PARTIAL_RESULT_EXPORTED_ORIGINAL_LOOP_FALLBACK_ONCE',
            body_def_use_ref=_reference('canonical_execution_plan', '#/region'),
            body_effect_order_ref=_reference('canonical_execution_plan', '#/proof/effects'),
            device_difference_removes_dependencies=False),
        Q=dict(status='CONDITIONAL_EXACT_SCALAR_DAG', scalar_dtype='float64',
            arithmetic_order=canonical.proof['arithmetic_order'],
            original_call_order=certificate['semantics']['call_order'],
            reassociation=False, commutation=False, contraction=False, fast_math=False,
            algorithm_invariants=['ORIGINAL_ALGORITHM_DATAFLOW', 'UNCHANGED_FLOAT64_DTYPE',
                'EXACT_ORIGINAL_ITERATION_COUNT', 'NO_QUANTIZATION',
                'NO_APPROXIMATE_OPERATOR_SUBSTITUTION', 'NO_ITERATION_DELETION'],
            implementation_scope='CURRENT_STRICT_CPU_CONTRACT_NOT_A_CROSS_DEVICE_EQUIVALENCE_PROOF',
            floating_environment_policy=report['floating_environment_policy'],
            full_numpy_status_flag_equivalence=False,
            failure_policy=report['failure_policy']),
        effects=dict(body=deepcopy(canonical.proof['effects']),
            intermediate_values_escape=False, scope='ADMITTED_LOOP_AND_BODY_ONLY',
            whole_program_effects_closed=False),
        guards=dict(body_component=list(canonical.proof['runtime_guards']),
            loop=list(report['runtime_guards']), builtin_range='EXACT_BUILTIN_RANGE_IDENTITY',
            callables_ref='#/source/callee_guards', live_guard_discharge_proven=False),
        external_assumptions=list(report['external_assumptions']),
        provenance=dict(loop_origins_ref=_reference('loop', '#/origins'),
            body_origins_ref=_reference('loop', '#/body/source/origins'),
            fusion_certificate=dict(sha256=semantic_sha256(certificate),
                ref=_reference('strategy', '#/body_fusion_certificate')),
            raw_region_sha256=report['body_raw_region_sha256'],
            optimized_region=dict(id=canonical.region.id,
                sha256=semantic_sha256(plan_document(canonical)['region']),
                nodes=len(canonical.region.nodes), output=canonical.region.output),
            source_operation_map=source_map, calls=calls,
            storage_ids_used_as_logical_identity=False,
            original_graph_mutated=False, loop_iterations_materialized=False),
        scope='DERIVED_PROFILE_BOUNDARY_NOT_WHOLE_PROGRAM_CLOSURE')
    value['id'] = _PREFIX + boundary_digest(value)
    return value


def verify_loop_boundary(loop, strategy, plan, boundary):
    """Pure replay; a matching digest alone is never legality evidence."""
    expected = build_loop_boundary(loop, strategy, plan)
    if expected != boundary:
        raise ValueError('loop semantic boundary differs from original complete replay')
    return expected


__all__ = ['build_loop_boundary', 'verify_loop_boundary', 'boundary_digest']

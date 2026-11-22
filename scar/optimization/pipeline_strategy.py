"""Payload-only representation choice: fixed host pool or certified full fusion."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from scar.compiler.model import document as plan_document
from scar.compiler.pipeline_frontend import pipeline_digest, read_pipeline, validate_pipeline
from scar.compiler.validation import validate_plan
from scar.ir.record_codec import loads
from .optimizer import optimize_region
from scar.optimization.pipeline_memory import plan_pipeline_memory
from .resource_solver import _shape_instantiation, semantic_sha256


def _pipeline(value):
    return read_pipeline(value) if isinstance(value, (str, Path)) else validate_pipeline(value)


def _fusion(pipeline, budget):
    from .pipeline_fusion import fuse_pipeline_region, verify_pipeline_fusion

    raw, certificate = fuse_pipeline_region(pipeline)
    verify_pipeline_fusion(pipeline, raw, certificate)
    if raw is None:
        return None, certificate, None
    plan = optimize_region(raw, memory_budget_bytes=budget)
    if plan.proof.get('supported') is not True or plan.proof.get('legal') is not True:
        raise ValueError('optimized fused region is outside the guarded CPU profile')
    validate_plan(plan)
    return raw, certificate, plan


def _shape_replay(pipeline, raw, plan, certificate, preserve):
    issues, rows = [], []
    if raw.parameters != pipeline.parameters or plan.region.parameters != pipeline.parameters:
        issues.append(dict(field='fusion.parameters', reason='all original parameter shape obligations must survive'))
    parameters = preserve['input_shapes']
    # Replay the original logical value shapes through their certified raw-DAG
    # destinations using the existing ABI shape authority. No physical buffer
    # slot or optimized CSE alias is mistaken for an original logical value.
    for value in preserve['values']:
        expression = certificate.get('value_map', {}).get(value['id'])
        if expression is None:
            issues.append(dict(field='fusion.value_map.' + value['id'], reason='original value has no merged expression'))
            continue
        _, shape, errors = _shape_instantiation(replace(plan, region=replace(raw, output=expression)), parameters)
        actual = list(shape) if shape is not None else None
        if errors or actual is None or actual != value['shape']:
            issues.append(dict(field='fusion.shapes.' + value['id'], reason='original and merged value shapes differ or remain unknown'))
        rows.append(dict(value_id=value['id'], raw_expression=expression,
                         original_shape=value['shape'], merged_shape=actual))
    _, optimized_shape, errors = _shape_instantiation(plan, parameters)
    final = next(value for value in preserve['values'] if value['id'] == pipeline.output)
    if errors or optimized_shape is None or list(optimized_shape) != final['shape']:
        issues.append(dict(field='fusion.optimized_output_shape', reason='optimized output differs from original leaf composition'))
    return rows, issues


def plan_pipeline_strategy(pipeline, *, input_shapes, memory_budget_bytes, search_budget=100000):
    """Choose only within two explicit representations and the payload objective.

    Original callees and their guards remain in the PipelinePlan. A failing pool
    budget is not a rejection of a fused representation. No duration, DRAM or
    cache traffic, hardware placement, or partial fusion partition is inferred.
    """
    pipeline = _pipeline(pipeline)
    preserve = plan_pipeline_memory(pipeline, input_shapes=input_shapes,
                                    memory_budget_bytes=memory_budget_bytes, search_budget=search_budget)
    final = next(value for value in preserve['values'] if value['id'] == pipeline.output)
    raw, certificate, plan = None, None, None
    fusion_issues, shape_rows = [], []
    try:
        raw, certificate, plan = _fusion(pipeline, memory_budget_bytes)
        if raw is not None:
            shape_rows, fusion_issues = _shape_replay(pipeline, raw, plan, certificate, preserve)
    except (ValueError, TypeError, KeyError) as error:
        fusion_issues.append(dict(field='fusion.execution_profile', reason=str(error)))
        if certificate is None:
            from .pipeline_fusion import fuse_pipeline_region
            raw, certificate = fuse_pipeline_region(pipeline)
    if certificate is not None and not certificate.get('supported'):
        fusion_issues.extend(certificate.get('frontiers', ()))
    if preserve['frontier']:
        fusion_issues.extend(dict(field='original.' + issue['field'], reason=issue['reason'])
                             for issue in preserve['frontier'])
    reduction = dict(original_region_calls=len(pipeline.calls), fused_native_kernels=1,
                     original_array_materializations=len(pipeline.calls),
                     fused_array_materializations=1,
                     eliminated_intermediate_materializations=max(0, len(pipeline.calls) - 1))
    useful = reduction['eliminated_intermediate_materializations'] > 0
    fits = (type(memory_budget_bytes) is int and final['nbytes'] is not None
            and final['nbytes'] <= memory_budget_bytes)
    fusion_feasible = raw is not None and plan is not None and not fusion_issues and fits
    kind, selected = 'NO_OP', None
    if fusion_feasible and useful:
        kind = 'FULL_FUSION'
        selected = dict(schema='scar.fused-pipeline-memory-plan', schema_version=1,
                        pipeline_sha256=pipeline_digest(pipeline),
                        input_shapes=preserve['input_shapes'], output_shape=final['shape'],
                        final_bytes=final['nbytes'], total_workspace_bytes=0,
                        peak_allocation_bytes=final['nbytes'], memory_budget_bytes=memory_budget_bytes,
                        budget_satisfied=True, status='OPTIMAL_WITHIN_REPRESENTATION_PAYLOAD_MODEL', frontier=[],
                        memory_policy='UNBOUND', execution_representation='FULL_FUSION',
                        allocation_policy='ONE_FRESH_CALLER_OWNED_FINAL_OUTPUT_NO_WORKSPACE',
                        logical_intermediate_values_materialized=False,
                        budget_scope=preserve['budget_scope'], budget_excludes=preserve['budget_excludes'],
                        physical_optimality_proven=False, duration_predicted=False)
    elif preserve['status'] == 'OPTIMAL_WITHIN_FIXED_POOL_MODEL' and preserve['budget_satisfied']:
        kind, selected = 'PRESERVE_CALLS_HOST_POOL', preserve
    if raw is not None and not useful:
        fusion_issues.append(dict(field='fusion.reduction', reason='no intermediate materialization or call graph reduction'))
    if raw is not None and not fits:
        fusion_issues.append(dict(field='fusion.memory_budget_bytes', reason='fresh final payload does not fit a known budget'))
    best_bytes = None if selected is None else selected['peak_allocation_bytes']
    lower = final['nbytes']
    reaches_bound = best_bytes is not None and lower is not None and best_bytes == lower
    return dict(
        schema='scar.pipeline-strategy-plan', schema_version=1, pipeline_id=pipeline.id,
        pipeline_sha256=pipeline_digest(pipeline), input_shapes=preserve['input_shapes'],
        memory_budget_bytes=memory_budget_bytes, search_budget=search_budget,
        status='SELECTED_UNDER_MODEL_CONTRACT' if selected is not None else 'NO_NATIVE_STRATEGY_ADMITTED',
        selected_strategy=kind, selected=dict(kind=kind), selected_memory_plan=selected,
        preserve_memory_plan=preserve, fusion_certificate=certificate,
        fusion_region_sha256=None if certificate is None else certificate.get('region_sha256'),
        fusion_execution_plan_sha256=None if plan is None else semantic_sha256(plan_document(plan)),
        fusion_shape_replay=shape_rows, fusion_frontier=fusion_issues,
        reduction=reduction, output_shape=final['shape'], final_bytes=final['nbytes'],
        payload_lower_bound_bytes=lower, selected_payload_bytes=best_bytes,
        payload_gap_bytes=None if best_bytes is None or lower is None else best_bytes - lower,
        representation_optimality='REPRESENTATION_PAYLOAD_MODEL_OPTIMUM_ESTABLISHED' if reaches_bound else
                                  'FIXED_POOL_MODEL_ONLY' if kind == 'PRESERVE_CALLS_HOST_POOL' else 'NOT_ESTABLISHED',
        optimality_scope='GIVEN_FRESH_FINAL_Q_AND_TWO_SUPPORTED_REPRESENTATIONS_INCREMENTAL_PAYLOAD_ONLY',
        graph_authority='ORIGINAL_PIPELINE_AND_SHARED_LEAF_PLANS_RETAINED',
        provenance_chain=dict(original_to_raw='fusion_certificate.calls[*].source_expr_map',
                              raw_to_optimized='fused_execution_plan(...).proof.source_operation_map',
                              inferred_content_versions=False),
        frontier=[] if selected is not None else preserve['frontier'],
        original_python_no_op=dict(available=True, native_budget_proven=False),
        physical_optimality_proven=False, placement_optimized=False, partial_partition_search=False,
        duration_predicted=False, dram_traffic_predicted=False, target_variants_executed=False,
        deployment_authorized=False)


def verify_pipeline_strategy(pipeline, report):
    """Pure replay of original pool, raw certificate, optimized DAG and shapes."""
    value = loads(Path(report).read_bytes()) if isinstance(report, (str, Path)) else deepcopy(report)
    if (value.get('schema'), value.get('schema_version')) != ('scar.pipeline-strategy-plan', 1):
        raise ValueError('versioned pipeline strategy required')
    expected = plan_pipeline_strategy(pipeline, input_shapes=value['input_shapes'],
        memory_budget_bytes=value['memory_budget_bytes'], search_budget=value['search_budget'])
    if semantic_sha256(expected) != semantic_sha256(value):
        raise ValueError('pipeline strategy differs from complete representation/source/shape replay')
    return value


def fused_execution_plan(pipeline, strategy):
    """Return the exact optimized plan; its raw source certificate stays separate."""
    pipeline = _pipeline(pipeline)
    report = verify_pipeline_strategy(pipeline, strategy)
    if report['selected_strategy'] != 'FULL_FUSION':
        raise ValueError('full fusion is not the selected supported strategy')
    _, _, plan = _fusion(pipeline, report['memory_budget_bytes'])
    if plan is None or semantic_sha256(plan_document(plan)) != report['fusion_execution_plan_sha256']:
        raise ValueError('optimized fusion execution-plan receipt differs')
    return plan


__all__ = ['plan_pipeline_strategy', 'verify_pipeline_strategy', 'fused_execution_plan']

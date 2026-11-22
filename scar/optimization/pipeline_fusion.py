"""Order-preserving scalar DAG composition, guarded by the original pipeline.

This module creates a derived region over existing leaf expressions. It does
not select Python functions, execute values, infer shapes or discharge the
original global-callee/type/floating-status guards.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json

from scar.compiler.model import ArrayRegion,Expr,document as leaf_document
from scar.compiler.pipeline_frontend import PipelinePlan, document as pipeline_document, validate_pipeline
from scar.compiler.model import validate_region
from scar.compiler.model import COPY_REQUIRED_GUARDS


SCHEMA='scar.pipeline-fusion'
_ARITHMETIC=frozenset(('add','sub','mul','div','neg'))
_ALLOWED=_ARITHMETIC|{'input','const','copy'}


def _canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def _sha(value):return sha256(_canonical(value).encode()).hexdigest()


def fusion_region_digest(region):
    """Hash the complete exact region contract, including source provenance."""
    if type(region)is not ArrayRegion:raise ValueError('typed fused ArrayRegion required')
    return _sha(asdict(region))


def _frontier(certificate,field,reason,**detail):
    certificate['frontiers'].append(dict(field=field,reason=reason,**detail))


def _certificate(pipeline,wire):
    return dict(schema=SCHEMA,schema_version=1,supported=False,status='UNSUPPORTED_FULL_PIPELINE_FUSION',
        pipeline_sha256=_sha(wire),pipeline_source_ref='#/source',
        pipeline_source_sha256=_sha(wire['source']),region_sha256=None,frontiers=[],
        leaf_contracts={key:dict(plan_ref='#/leaf_plans/'+key,plan_sha256=_sha(plan),
            source_ref='#/leaf_plans/'+key+'/region/source',source_sha256=_sha(plan['region']['source']))
            for key,plan in wire['leaf_plans'].items()},calls=[],input_map={},value_map={},
        original_pipeline_preserved=True,original_pipeline_guards_required=True,
        standalone_region_installation_authorized=False,synthetic_callee_created=False,
        logical_content_versions_inferred=False,target_executed=False,
        shape_contract=dict(status='REPLAY_ORIGINAL_LEAF_AND_MERGED_DAGS_REQUIRED',
            external_parameter_shapes_assumed_equal=False,
            original_parameter_and_final_output_obligations_retained=True),
        semantics=dict(call_order='ORIGINAL_SOURCE_ORDER',per_element_operation_order='PRESERVED',
            operand_order='PRESERVED',reassociation=False,common_subexpression_elimination=False,
            calls_omitted=False,physical_memory_traffic_predicted=False,profitability_proven=False))


def fuse_pipeline_region(pipeline):
    """Return an exact merged candidate or explicit unsupported frontiers.

    Shared leaf plans stay in the original PipelinePlan. A certificate uses
    their immutable document hashes and JSON pointers, never copies their
    graphs or invents global bindings. Every original call must contribute its
    arithmetic/materialization to the final dependency closure; dead calls are
    never erased just because they contain no arithmetic instructions.
    """
    try:
        validate_pipeline(pipeline);wire=pipeline_document(pipeline)
    except ValueError as error:
        if type(pipeline)is not PipelinePlan:raise ValueError('typed PipelinePlan required') from error
        return None,dict(schema=SCHEMA,schema_version=1,supported=False,
            status='UNSUPPORTED_FULL_PIPELINE_FUSION',pipeline_sha256=_sha(asdict(pipeline)),
            pipeline_hash_basis='REJECTED_TYPED_CONTRACT',region_sha256=None,
            frontiers=[dict(field='pipeline.contract',reason='PIPELINE_CONTRACT_REJECTED',detail=str(error))],
            original_pipeline_preserved=True,target_executed=False)
    certificate=_certificate(pipeline,wire)
    for index,call in enumerate(pipeline.calls):
        effects=call.plan.proof.get('effects',{})
        if (effects.get('input_writes')is not False or effects.get('output_policy')!='fresh_each_call'
                or effects.get('purity_basis')!='closed_operator_semantics'):
            _frontier(certificate,'calls.'+str(index)+'.effects','LEAF_CLOSED_EFFECT_CONTRACT_REQUIRED',call_id=call.id)
        for field in ('escapes','external_effect','rng_effect'):
            if field in effects and effects[field] not in (False,(),[]):
                _frontier(certificate,'calls.'+str(index)+'.effects.'+field,
                    'INTERMEDIATE_ESCAPE_OR_EFFECT_OPEN',call_id=call.id)
        for node in call.plan.region.nodes:
            if node.op not in _ALLOWED:
                _frontier(certificate,'calls.'+str(index)+'.expressions.'+node.id,
                    'SLICE_OR_NON_ELEMENTWISE_OPERATION',call_id=call.id,operation=node.op)
    if certificate['frontiers']:return None,certificate

    nodes=[];values={};arithmetic_by_call=[];copies_by_call=[];copy_sources=[]
    for index,(parameter,value) in enumerate(zip(pipeline.parameters,pipeline.input_values,strict=True)):
        identity='fused:input:'+str(index)
        nodes.append(Expr(identity,'input',(),dict(name=parameter,index=index)))
        values[value]=identity
    certificate['input_map']=dict(values)
    for index,(call,call_wire) in enumerate(zip(pipeline.calls,wire['calls'],strict=True)):
        parameters=dict(zip(call.plan.region.parameters,call.inputs,strict=True))
        mapping={};arithmetic=[];copies=[]
        for ordinal,node in enumerate(call.plan.region.nodes):
            if node.op=='input':
                mapping[node.id]=values[parameters[node.attrs['name']]]
            else:
                identity='fused:call:'+str(index)+':expr:'+str(ordinal)
                nodes.append(Expr(identity,node.op,tuple(mapping[value] for value in node.inputs),deepcopy(node.attrs)))
                mapping[node.id]=identity
                if node.op in _ARITHMETIC:arithmetic.append(identity)
                if node.op=='copy':copies.append(identity)
        values[call.output]=mapping[call.plan.region.output]
        source_mapping={}
        for expression,route in call.plan.proof.get('source_operation_map',{}).items():
            if not isinstance(route,dict):
                _frontier(certificate,'calls.'+str(index)+'.source_operation_map',
                    'SOURCE_EXPRESSION_MAPPING_OPEN',call_id=call.id,expression=expression);continue
            target=route.get('optimized_node')
            if target is not None and target not in mapping:
                _frontier(certificate,'calls.'+str(index)+'.source_operation_map',
                    'SOURCE_EXPRESSION_DESTINATION_MISSING',call_id=call.id,expression=expression);continue
            source_mapping[expression]=None if target is None else mapping[target]
        plan_id=call_wire['plan_id']
        certificate['calls'].append(dict(call_id=call.id,call_ref='#/calls/'+str(index),
            call_sha256=_sha(call_wire),leaf_plan_id=plan_id,leaf_plan_ref='#/leaf_plans/'+plan_id,
            leaf_plan_sha256=certificate['leaf_contracts'][plan_id]['plan_sha256'],
            expr_map=mapping,source_expr_map=source_mapping,output_value=call.output,
            output_expr=values[call.output]))
        arithmetic_by_call.append(arithmetic)
        copies_by_call.append(copies)
        source_copies=[dict(expression=expression,route=deepcopy(route)) for expression,route
            in call.plan.proof.get('source_operation_map',{}).items()
            if isinstance(route,dict) and route.get('source_op')=='copy']
        if copies and not source_copies:
            _frontier(certificate,'calls.'+str(index)+'.source_operation_map',
                'ORIGINAL_COPY_WORK_MAPPING_REQUIRED',call_id=call.id)
        if source_copies:
            if call.plan.cost.get('baseline_copy_materializations')!=len(source_copies):
                _frontier(certificate,'calls.'+str(index)+'.cost',
                    'ORIGINAL_COPY_WORK_ACCOUNTING_DIFFERS',call_id=call.id)
            copy_sources.append(dict(call_id=call.id,leaf_plan_ref='#/leaf_plans/'+plan_id,
                source_copies=source_copies,source_copy_materializations=len(source_copies),
                represented_copy_expressions=copies))
    certificate['value_map']=dict(values)
    if certificate['frontiers']:return None,certificate
    by_id={node.id:node for node in nodes};needed=set();pending=[values[pipeline.output]]
    while pending:
        identity=pending.pop()
        if identity in needed:continue
        needed.add(identity);pending.extend(by_id[identity].inputs)
    for index,(call,arithmetic) in enumerate(zip(pipeline.calls,arithmetic_by_call,strict=True)):
        missing=[identity for identity in arithmetic if identity not in needed]
        if missing:
            _frontier(certificate,'calls.'+str(index),'ORIGINAL_CALL_ARITHMETIC_OUTSIDE_FINAL_DEPENDENCY',
                call_id=call.id,expressions=missing)
        missing_copies=[identity for identity in copies_by_call[index] if identity not in needed]
        if missing_copies:
            _frontier(certificate,'calls.'+str(index),'ORIGINAL_CALL_MATERIALIZATION_OUTSIDE_FINAL_DEPENDENCY',
                call_id=call.id,expressions=missing_copies)
    if certificate['frontiers']:return None,certificate
    assumptions=('EXACT_NUMPY_NDARRAY_FLOAT64_CPU_2D','FRESH_OUTPUT','NO_INPUT_WRITES',
         'ORIGINAL_PIPELINE_CALLEE_GUARDS_REQUIRED','ORIGINAL_PIPELINE_SHAPE_OBLIGATIONS_REQUIRED')
    if copy_sources:
        assumptions+=tuple(guard for guard in COPY_REQUIRED_GUARDS if guard not in assumptions)
        certificate['copy_work']=dict(source_calls=copy_sources,
            original_materialization_count=sum(row['source_copy_materializations'] for row in copy_sources),
            represented_copy_expression_count=sum(len(row) for row in copies_by_call),
            labels=['MEM','XFER','REP'],logical_value_not_object_or_storage_equivalence=True,
            actual_physical_transfer='UNKNOWN_NOT_RUNTIME_OBSERVED',
            equivalence_scope='NORMAL_COMPLETION_NOT_INTERNAL_ALLOCATION_FAILURE_SEQUENCE',
            final_fresh_output_required=True,profitability_proven=False)
    region=ArrayRegion('fused-pipeline:'+certificate['pipeline_sha256'][:24],pipeline.source,
        pipeline.parameters,tuple(nodes),values[pipeline.output],
        assumptions)
    validate_region(region)
    certificate.update(supported=True,status='GUARDED_FULL_PIPELINE_FUSION_CANDIDATE',
        region_sha256=fusion_region_digest(region),region_id=region.id,
        merged_expression_count=len(nodes),arithmetic_expression_count=sum(len(row) for row in arithmetic_by_call))
    return region,certificate


def replay_pipeline_fusion(pipeline, certificate):
    """Return the fused body after checking the retained original-edge certificate."""
    region, expected = fuse_pipeline_region(pipeline)
    if _canonical(certificate) != _canonical(expected):
        raise ValueError('pipeline fusion certificate differs from complete source/leaf/call replay')
    return region


def verify_pipeline_fusion(pipeline,region,certificate):
    """Replay exact original call/leaf/source contracts and every mapped edge."""
    expected_region = replay_pipeline_fusion(pipeline, certificate)
    if expected_region is None:
        if region is not None:raise ValueError('unsupported pipeline fusion cannot have a region')
    elif region is None or fusion_region_digest(region)!=fusion_region_digest(expected_region):
        raise ValueError('fused region differs from original operation/operand-order replay')
    return True


__all__ = ['fuse_pipeline_region', 'replay_pipeline_fusion', 'verify_pipeline_fusion', 'fusion_region_digest']

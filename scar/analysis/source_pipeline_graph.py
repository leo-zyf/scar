"""Source pipeline graph."""

from copy import deepcopy
import json
from scar.ir.record_codec import encode, decode
from scar.ir.v2.common import EvidenceClaim, EvidenceKind, ProofClaim, ProofStatus
from scar.ir.v2.contracts import (
    EffectSummary,
    ContractDefinition,
    ContractFacet,
    EffectTarget,
    EffectTargetKind,
    FacetRequirement,
)
from scar.ir.v2.ids import (
    OperationDefinitionID,
    ValueSlotID,
    ContractID,
    EffectSummaryID,
    OptimizationRegionID,
    PlanAlternativeID,
    PlanID,
)
from scar.ir.v2.semantic import (
    SemanticGraph,
    OperationDefinition,
    OperationKind,
    ValueSlot,
    SemanticEdge,
    SemanticEndpoint,
    SemanticNodeKind as N,
    SemanticRelation as R,
)
from scar.optimization.resource_solver import semantic_sha256
from scar.compiler.pipeline_frontend import (
    document as pipeline_document,
    pipeline_digest,
    validate_pipeline,
)
from scar.ir.program.caller_pipeline_values import CallerValueProjection
from scar.ir.program.dependencies import DependencyFrame
from scar.ir.program.providers import ProviderFactReference
from scar.ir.v2.codec import optimization_from_dict
from scar.ir.v2.optimization import (
    GuardSpec,
    InstructionKind,
    OptimizationContext,
    OptimizationGraph,
    OptimizationRegion,
    PlanAlternative,
    PlanDisposition,
    PlanInstruction,
    PlanSelection,
    RegionGranularity,
    RegionPort,
    RegionPortKind,
    ResidualEffect,
    TransformDelta,
    TransformKind,
    ValidationRequest,
    ValuePattern,
)
from scar.optimization.pipeline_strategy import plan_pipeline_strategy


# Project admitted caller routes into the existing static SemanticGraph.
#
# Slots denote original source routes, not runtime logical values, objects or
# allocations. Call sites read upstream slots through edges; their formal slot
# lists never claim ownership of another definition's public interface.
def project_source_pipeline_graph(pipeline, region_document, source_values):
    from scar.analysis.source_pipeline_graph import read_fusion_region, _contract
    from scar.compiler.pipeline_frontend import pipeline_digest

    oir=read_fusion_region(region_document)
    qualification=region_document['qualification'];projection=region_document['projection']
    if (source_values.pipeline_sha256!=pipeline_digest(pipeline)
            or source_values.pipeline_sha256!=qualification['pipeline_sha256']
            or semantic_sha256(encode(source_values))!=qualification['source_values_sha256']):
        raise ValueError('source pipeline/value routes differ from admitted region')
    region=next(iter(oir.regions.values()))
    root=OperationDefinitionID('source-region:'+region.id.value)
    graph=SemanticGraph()
    graph.add_contract(_contract(region.contract,qualification))
    graph.add_effect(EffectSummary(region.effect_summary))
    graph.add_definition(OperationDefinition(root,OperationKind.REGION,'Source-qualified caller region',
        code_id=pipeline.source.code_key,source_file=pipeline.source.path,
        contract=region.contract,effect_summary=region.effect_summary,
        metadata=dict(source=encode(pipeline.source),original_region_id=region.id.as_dict(),
            boundary_ports=deepcopy(projection['boundary_ports']),
            world_routes=deepcopy(projection['world_routes']),
            Q=deepcopy(qualification['Q']),runtime_values_created=False,
            scope='ORIGINAL_SOURCE_CALLER_NORMAL_COMPLETION_UNDER_RETAINED_GUARDS')))
    evidence=EvidenceClaim(EvidenceKind.INFERRED,
        references=tuple(json.dumps(row,sort_keys=True,separators=(',',':'))
                         for row in region_document['source_refs']),scope=region.id.wire,
        assumptions=tuple(qualification['required_runtime_guards']))
    def edge(relation,source,source_kind,target,target_kind,role):
        identity='source-edge:'+semantic_sha256(dict(relation=relation.value,
            source=source.wire,target=target.wire,role=role))
        graph.add_edge(SemanticEdge(identity,relation,SemanticEndpoint(source_kind,source),
            SemanticEndpoint(target_kind,target),evidence))
    def slot(role,name,owner,routes,axis):
        identity=ValueSlotID('source-route:'+semantic_sha256(dict(
            pipeline=pipeline_digest(pipeline),source_parents=projection['source_parents'],role=role,routes=routes)))
        graph.add_slot(ValueSlot(identity,name,role,semantic_type='source-'+axis+'-route',owner=owner,
            metadata=dict(axis=axis,source_routes=deepcopy(routes),
                runtime_logical_version=False,storage_identity_inferred=False)))
        edge(R.CONTAINS,owner,N.OPERATION,identity,N.VALUE_SLOT,'slot-owner')
        return identity
    boundary={};values={}
    for port in projection['boundary_ports']:
        identity=slot(port['port_id'],port['name'],root,port['routes'],
            'data' if port['kind'] in ('input','output') else 'world')
        boundary[port['port_id']]=identity
        if port['kind'] in ('input','output'):values[port['value']]=identity
    graph.definitions[root].input_slots=tuple(boundary[row['port_id']] for row in projection['boundary_ports'] if row['kind']=='input')
    graph.definitions[root].output_slots=tuple(boundary[row['port_id']] for row in projection['boundary_ports'] if row['kind']=='output')
    graph.definitions[root].state_slots=tuple(boundary[row['port_id']] for row in projection['boundary_ports'] if row['kind']=='state')
    for source,target,relation in ((root,region.contract,R.HAS_CONTRACT),(root,region.effect_summary,R.HAS_EFFECT)):
        edge(relation,source,N.OPERATION,target,N.CONTRACT if relation is R.HAS_CONTRACT else N.EFFECT,'region-Q')
    calls={row.id:row for row in source_values.calls};previous=None;members=[]
    for index,(call,record) in enumerate(zip(pipeline.calls,projection['operations'],strict=True)):
        route=calls[call.id]
        if route.index!=index or route.inputs!=call.inputs or route.output!=call.output:
            raise ValueError('original call route order or operands differ')
        identity=OperationDefinitionID(record['projected_id']['value']);members.append(identity)
        graph.add_definition(OperationDefinition(identity,OperationKind.OPERATOR,'Static call '+str(index),
            source_file=pipeline.source.path,parent_id=root,contract=region.contract,
            effect_summary=region.effect_summary,metadata=dict(
                kind='ORIGINAL_STATIC_CALL_SITE_NOT_DYNAMIC_INVOCATION',
                original_operation=deepcopy(record),source_body=encode(route.shared_body),
                ordered_arguments=encode(route.substitutions),source_result=encode(route.source_result),
                source_world_before=encode(route.world_before),source_world_after=encode(route.world_after))))
        edge(R.CONTAINS,root,N.OPERATION,identity,N.OPERATION,'body-member')
        edge(R.HAS_CONTRACT,identity,N.OPERATION,region.contract,N.CONTRACT,'call-Q')
        edge(R.HAS_EFFECT,identity,N.OPERATION,region.effect_summary,N.EFFECT,'call-effects-remain-unknown')
        for ordinal,(value,substitution) in enumerate(zip(call.inputs,route.substitutions,strict=True)):
            if value!=substitution.value or value not in values:
                raise ValueError('source consumer precedes its admitted producer')
            edge(R.CONSUMES,identity,N.OPERATION,values[value],N.VALUE_SLOT,'argument:'+str(ordinal))
        if call.output not in values:
            values[call.output]=slot('result:'+str(index),call.output,identity,
                dict(frame=encode(route.frame),source_result=encode(route.source_result)), 'data')
            graph.definitions[identity].output_slots=(values[call.output],)
        edge(R.PRODUCES,identity,N.OPERATION,values[call.output],N.VALUE_SLOT,'call-result')
        world=[]
        for side,original in (('before',route.world_before),('after',route.world_after)):
            public='world:entry' if side=='before' and index==0 else 'world:exit' if side=='after' and index==len(pipeline.calls)-1 else None
            if public is None:
                endpoint=slot('world:'+str(index)+':'+side,side,identity,
                    dict(frame=encode(route.frame),original_port=encode(original)), 'world')
                world.append(endpoint)
            else:
                endpoint=boundary[public]
                port=next(row for row in projection['boundary_ports'] if row['port_id']==public)
                if port['routes']['call_world']!=encode(original):
                    raise ValueError('public world endpoint differs from exact original occurrence')
            edge(R.READS_SLOT if side=='before' else R.WRITES_SLOT,identity,N.OPERATION,
                endpoint,N.VALUE_SLOT,'original-world-'+side)
        graph.definitions[identity].state_slots=tuple(world)
        if previous is not None:
            edge(R.CONTROLS,previous,N.OPERATION,identity,N.OPERATION,
                'ORIGINAL_CALLER_SOURCE_ORDER_NOT_WORLD_VALUE_EQUALITY')
        previous=identity
    graph.assert_valid()
    return graph,dict(root=root.as_dict(),members=[item.as_dict() for item in members],
        boundary_slots={key:value.as_dict() for key,value in boundary.items()},
        data_slots={key:value.as_dict() for key,value in values.items()},
        world_endpoint_identity='EXACT_SOURCE_OCCURRENCE_NO_ADJACENT_WORLD_MERGE')


# Project an admitted F4 pipeline into the existing report-only v2 OIR.
#
# The original source bodies stay in their parents.  Projected CALL definitions
# are explicit adapter identities, not aliases for ProgramOperationIDs.  Global
# ports remain source-route constraints, never fabricated runtime value versions.
# Source admission and live guard enforcement belong to the enclosing producer.
SCHEMA = 'scar.pipeline-region-pass'
ORIGINAL = 'ORIGINAL_PYTHON_NO_OP'
FUSION = 'FULL_FUSION'
_PROJECTION = 'PROGRAM_CALL_TO_V2_REGION_DEFINITION_V1'
_WORLD = 'UNKNOWN_OUTSIDE_GUARDED_CLOSED_ARRAY_PROFILE'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _proof(value):
    from scar.compiler.caller_pipeline import CallerPipelineProof, document as proof_document
    if type(value) is CallerPipelineProof:
        return proof_document(value)
    _need(type(value) is dict, 'original caller pipeline proof required')
    if value.get('schema') == 'scar.caller-matched-array-pipeline-proof':
        _need(value.get('schema_version') == 1, 'unsupported caller pipeline proof')
        return deepcopy(value)
    return proof_document(decode(CallerPipelineProof, value))


def _projected_id(row):
    return OperationDefinitionID('pipeline-call-projection:' + semantic_sha256(dict(
        rule=_PROJECTION, namespace=row['caller_namespace'],
        program_operation=row['program_operation'], original_fact_ref=row['original_fact_ref'])))


def _pattern(routes, *, role, value=None):
    constraints = [('scope', 'SOURCE_TEMPLATE_NOT_RUNTIME_LOGICAL_OR_STORAGE_IDENTITY'),
        ('role', role), ('original_global_ports', _json(routes))]
    if value is not None:
        constraints.append(('pipeline_value_label', value))
    return ValuePattern(semantic_type='original-source-port-route', constraints=tuple(constraints))


def _boundary(proof, values, pipeline):
    calls = {row.id: row for row in values.calls}
    origin = proof['caller_origin']
    formals = origin['caller_parameters'][1:-1]
    _need(len(formals) == len(pipeline.parameters), 'original caller input boundary differs')
    ports = []
    for index, (name, value, formal) in enumerate(zip(
            pipeline.parameters, pipeline.input_values, formals, strict=True)):
        actuals = [encode(route.actual) for call in values.calls
            for route in call.substitutions if route.value == value]
        ports.append(dict(port_id='input:' + str(index), kind='input', name=name,
            value=value, routes=dict(caller_formal=formal, call_actuals=actuals)))
    producer = next(call for call in pipeline.calls if call.output == pipeline.output)
    ports.append(dict(port_id='output:0', kind='output', name='result',
        value=pipeline.output, routes=dict(caller_return=origin['caller_return'],
            call_result=encode(calls[producer.id].source_result))))
    worlds = [dict(pipeline_call=call.id, before=encode(calls[call.id].world_before),
        after=encode(calls[call.id].world_after)) for call in pipeline.calls]
    ports.extend((dict(port_id='world:entry', kind='state', name='incoming world',
        value=None, routes=dict(caller_formal=origin['caller_parameters'][-1],
            call_world=worlds[0]['before'])),
        dict(port_id='world:exit', kind='state', name='normal outgoing world',
            value=None, routes=dict(caller_return=origin['caller_return_world'],
                call_world=worlds[-1]['after']))))
    return ports, worlds


def _graph(document):
    """Reconstruct the OIR from the explicit projection, never from names."""
    projection, q = document['projection'], document['qualification']
    members = tuple(_projected_id(row) for row in projection['operations'])
    _need(members and len(set(members)) == len(members), 'one projected member per original CALL required')
    for row, identity in zip(projection['operations'], members, strict=True):
        frame = decode(DependencyFrame, row['original_frame'])
        fact = decode(ProviderFactReference, row['original_fact_ref'])
        _need(row['projected_id'] == identity.as_dict()
            and row['program_operation'] == encode(frame.operation)
            and row['caller_namespace'] == encode(frame.caller)
            and row['source_parent_sha256'] == fact.artifact.sha256,
            'original Program CALL / FactRef / projected definition correspondence differs')
    rid = OptimizationRegionID('pipeline-composite:' + semantic_sha256(dict(
        pipeline=q['pipeline_sha256'], members=[item.wire for item in members],
        boundary=projection['boundary_ports'])))
    _need(document['region_id'] == rid.as_dict(), 'composite region identity differs from original members/boundary')
    cid = ContractID('pipeline-fusion-Q:' + semantic_sha256(q['Q']))
    eid = EffectSummaryID('pipeline-world:' + semantic_sha256(projection['world_routes']))
    graph = OptimizationGraph(OptimizationContext(definitions=frozenset(members),
        contracts=frozenset((cid,)), effects=frozenset((eid,))))
    for row in projection['boundary_ports']:
        graph.add_port(RegionPort(row['port_id'], RegionPortKind(row['kind']), row['name'],
            value=_pattern(row['routes'], role=row['port_id'], value=row['value'])))
    graph.add_port(RegionPort('world:effects', RegionPortKind.EFFECT, 'unclosed world effects',
        effect=EffectTarget(EffectTargetKind.OPAQUE, 'source-world:' + eid.wire)))
    graph.add_region(OptimizationRegion(rid, RegionGranularity.DATAFLOW_SUBGRAPH,
        'Complete original caller pipeline', definitions=members,
        ports=tuple(graph.ports), effect_summary=eid, contract=cid))
    guards = tuple(GuardSpec('pipeline-guard:' + semantic_sha256(guard), guard)
        for guard in q['required_runtime_guards'])
    original = PlanAlternativeID(rid.value + ':' + ORIGINAL)
    graph.add_alternative(PlanAlternative(original, rid, ORIGINAL, TransformKind.NO_OP,
        original=True))
    alternatives = {ORIGINAL: original.wire}
    strategy = document['strategy']
    if strategy.get('selected_strategy') == FUSION:
        certificate = strategy['fusion_certificate']
        _need(certificate.get('supported') is True and not strategy['fusion_frontier']
            and certificate['pipeline_sha256'] == q['pipeline_sha256'],
            'FULL_FUSION requires an admitted source/shape fusion strategy')
        identity = PlanAlternativeID(rid.value + ':' + FUSION)
        condition = 'Conditional when every listed runtime guard and explicit Q assumption holds.'
        evidence = EvidenceClaim(EvidenceKind.INFERRED,
            references=('#/strategy/fusion_certificate', '#/projection/source_parents', '#/qualification/Q'),
            scope=rid.wire, assumptions=tuple(q['required_runtime_guards']))
        graph.add_alternative(PlanAlternative(identity, rid, FUSION, TransformKind.FUSE,
            delta=TransformDelta(removed_definitions=members,
                added_operations=(projection['composite_operation'],)),
            proofs=(ProofClaim('ORIGINAL_CALL_OPERAND_ORDER_AND_GUARDED_BOUNDARY_PRESERVED',
                ProofStatus.PROVEN, (evidence,), condition),), guards=guards,
            residual_effects=(ResidualEffect(eid, condition + ' Unknown world effects remain explicit.',
                'PRESERVE_ORIGINAL_REGION_WORLD_PORTS'),),
            validation_request=ValidationRequest(cid, (1, 2, 3, 4),
                (q['source_proof_sha256'], q['source_values_sha256'])),
            instructions=(PlanInstruction('replace-composite-region', InstructionKind.REPLACE_VALUE_PATH,
                rid.wire, replacement=dict(kind=FUSION, operation=projection['composite_operation'],
                    fusion_region_sha256=certificate['region_sha256'],
                    preserved_port_ids=sorted(graph.ports)), rationale=condition),), fallback=original))
        alternatives[FUSION] = identity.wire
    _need(document['alternative_ids'] == alternatives, 'alternative kind/identity domain differs')
    _need(q['Q']['conditional_when'] == q['required_runtime_guards']
        and q['runtime_guards_discharged'] is False and q['world_effects'] == _WORLD
        and q['target_program_executed'] is False,
        'source projection cannot discharge guards, close world effects or claim execution')
    _need(q['contract'] == _contract(cid, q).as_dict()
        and q['effect_summary'] == EffectSummary(eid).as_dict(),
        'explicit Q or unknown effect summary differs')
    return graph


def _contract(identity, qualification):
    return ContractDefinition(identity, 'Original caller full-source guarded fusion Q',
        facets=(FacetRequirement(ContractFacet.OUTPUT, True,
                    description='Same public result, one fresh output, no input writes.'),
            FacetRequirement(ContractFacet.STATE, True,
                    description='Stable read-only inputs; original world routes retained; unclosed effects UNKNOWN.'),
            FacetRequirement(ContractFacet.NUMERICAL, True,
                    description='Original scalar operation and operand order; existing FP-status fallback.'),
            FacetRequirement(ContractFacet.EXCEPTION, True,
                    description='Existing normal-completion/fallback contract, not internal allocation-failure equivalence.'),
            FacetRequirement(ContractFacet.RESOURCE, True,
                    description='Explicit payload budget; no physical-time or global optimality claim.')),
        assumptions=tuple(qualification['required_runtime_guards']))


def derive_fusion_region(pipeline, memory, source_proof, source_values, *, source_refs=()):
    """Consume already replayed parents; perform only source/model composition.

    ``source_values`` is CallerValueProjection or its record-codec document.
    The parent producer must cold-replay the source proof before calling this.
    No source import, compilation, native load or target execution occurs here.
    """
    pipeline = validate_pipeline(pipeline)
    proof = _proof(source_proof)
    values = (source_values if type(source_values) is CallerValueProjection
        else decode(CallerValueProjection, source_values))
    _need(proof['pipeline'] == pipeline_document(pipeline)
        and values.pipeline_sha256 == pipeline_digest(pipeline), 'source proof/value/pipeline identity differs')
    _need(values.input_shapes == memory['input_shapes'], 'source value/memory shapes differ')
    _need([call.id for call in values.calls] == [call.id for call in pipeline.calls]
        and len({call.id for call in values.calls}) == len(pipeline.calls), 'source CALL route coverage differs')
    guards = list(values.required_guards)
    _need(guards and set(proof['required_runtime_guards']) <= set(guards), 'original source guards were discarded')
    _need(all(type(ref) is dict and ref for ref in source_refs), 'source references must be nonempty dictionaries')
    by_call = {row['pipeline_call']: row for row in proof['call_evidence']}
    _need(len(by_call) == len(proof['call_evidence']) == len(pipeline.calls), 'one original F4 witness per CALL required')
    operations = []
    for call, route in zip(pipeline.calls, values.calls, strict=True):
        witness = by_call[call.id]
        _need(encode(route.frame) == witness['frame']
            and encode(route.original_call) == witness['original_call']
            and route.inputs == call.inputs and route.output == call.output
            and encode(route.source_result) == witness['result']
            and encode(route.world_before) == witness['world_before']
            and encode(route.world_after) == witness['world_after'], 'CALL source/frame/value/world route differs')
        row = dict(pipeline_call=call.id, original_frame=encode(route.frame),
            program_operation=encode(route.frame.operation), caller_namespace=encode(route.frame.caller),
            original_fact_ref=encode(route.original_call), interface_fact_ref=encode(route.interface_fact),
            source_parent_sha256=route.original_call.artifact.sha256,
            original_body_ref=encode(route.shared_body), source_evidence_pointer=route.evidence_pointer)
        row['projected_id'] = _projected_id(row).as_dict()
        operations.append(row)
    boundary, worlds = _boundary(proof, values, pipeline)
    projection = dict(rule=_PROJECTION, operations=operations, boundary_ports=boundary,
        world_routes=worlds, composite_operation='pipeline-full-fusion:' + pipeline_digest(pipeline),
        source_parents=dict(source_proof_sha256=semantic_sha256(proof),
            source_values_sha256=semantic_sha256(encode(values)), source_code_sha256=pipeline.source.sha256,
            catalog=deepcopy(proof['catalog']), producer_pins=deepcopy(proof['producer_pins']),
            interface_manifest_sha256=semantic_sha256(proof['interface_manifest'])))
    strategy = plan_pipeline_strategy(pipeline, input_shapes=memory['input_shapes'],
        memory_budget_bytes=memory['memory_budget_bytes'], search_budget=memory.get('search_budget', 100000))
    final = next(row for row in memory['values'] if row['id'] == pipeline.output)
    q = dict(pipeline_sha256=pipeline_digest(pipeline), memory_plan_sha256=semantic_sha256(memory),
        source_proof_sha256=projection['source_parents']['source_proof_sha256'],
        source_values_sha256=projection['source_parents']['source_values_sha256'], source=encode(pipeline.source),
        input_shapes=deepcopy(values.input_shapes), output_shape=deepcopy(final['shape']),
        input_dtype='float64', input_layout='C_CONTIGUOUS', memory_budget_bytes=memory['memory_budget_bytes'],
        memory_budget_scope=memory['budget_scope'], required_runtime_guards=guards,
        normal_completion_only=True, runtime_guards_discharged=False, world_effects=_WORLD,
        source_bodies_copied=0, runtime_logical_versions_created=0, target_program_executed=False,
        source_admission='QUALIFIED_PARENT_F4_REPLAY_REQUIRED',
        frontier=deepcopy(strategy['fusion_frontier']))
    q['Q'] = dict(full_original_source_required=True, readonly_stable_inputs=True,
        original_fp_status_and_fallback_required=True, operand_order_preserved=True,
        fresh_output_required=True, no_input_writes=True, world_effects=_WORLD,
        equivalence_scope='GUARDED_NORMAL_COMPLETION_NOT_GENERAL_PYTHON_OR_INTERNAL_FAILURE_SEQUENCE',
        conditional_when=guards)
    members = [_projected_id(row).wire for row in operations]
    rid = OptimizationRegionID('pipeline-composite:' + semantic_sha256(dict(
        pipeline=q['pipeline_sha256'], members=members, boundary=boundary)))
    cid = ContractID('pipeline-fusion-Q:' + semantic_sha256(q['Q']))
    eid = EffectSummaryID('pipeline-world:' + semantic_sha256(worlds))
    q['contract'] = _contract(cid, q).as_dict()
    q['effect_summary'] = EffectSummary(eid).as_dict()
    alternatives = {ORIGINAL: PlanAlternativeID(rid.value + ':' + ORIGINAL).wire}
    if strategy['selected_strategy'] == FUSION:
        alternatives[FUSION] = PlanAlternativeID(rid.value + ':' + FUSION).wire
    result = dict(schema=SCHEMA, schema_version=1, region_id=rid.as_dict(), alternative_ids=alternatives,
        source_refs=deepcopy(list(source_refs)), projection=projection, qualification=q, strategy=strategy)
    result['optimization'] = _graph(result).to_dict()
    return result


def read_fusion_region(document):
    """Read the existing v2 codec with context reconstructed from provenance.

    This checks the adapter/OIR correspondence.  The enclosing producer still
    replays the original source parents and the complete fusion strategy.
    """
    _need(type(document) is dict and set(document) == {'schema', 'schema_version', 'region_id',
        'alternative_ids', 'source_refs', 'projection', 'qualification', 'optimization', 'strategy'}
        and document['schema'] == SCHEMA and document['schema_version'] == 1,
        'versioned pipeline fusion region pass document required')
    expected = _graph(document)
    actual = optimization_from_dict(document['optimization'], expected.context)
    _need(len(actual.plans) <= 1, 'one selection for the complete source region required')
    for plan in actual.plans.values():
        _need(plan.region == next(iter(expected.regions))
            and plan.disposition in (PlanDisposition.KEEP, PlanDisposition.REWRITE)
            and plan.selected is not None, 'complete-region KEEP or REWRITE selection required')
        expected.add_plan(plan)
    _need(expected.to_dict() == actual.to_dict(), 'OIR differs from original source region projection')
    return actual


def select_fusion_region(document, *, rewrite, reason):
    """Publish one KEEP/REWRITE selection without executing or deploying it."""
    _need(type(rewrite) is bool and type(reason) is str and reason,
        'explicit rewrite decision and reason required')
    result = deepcopy(document)
    graph = read_fusion_region(result)
    kind = FUSION if rewrite else ORIGINAL
    _need(kind in result['alternative_ids'], 'unsupported FULL_FUSION cannot be selected')
    selected = next(identity for identity in graph.alternatives
        if identity.wire == result['alternative_ids'][kind])
    region = next(iter(graph.regions))
    graph.plans.clear()
    graph.add_plan(PlanSelection(PlanID(region.value + ':selection'), region,
        PlanDisposition.REWRITE if rewrite else PlanDisposition.KEEP, selected, reason))
    result['optimization'] = graph.to_dict()
    return result


__all__ = [
    'SCHEMA',
    'ORIGINAL',
    'FUSION',
    'derive_fusion_region',
    'read_fusion_region',
    'select_fusion_region',
    'project_source_pipeline_graph',
]

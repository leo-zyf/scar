"""Project admitted caller routes into the existing static SemanticGraph.

Slots denote original source routes, not runtime logical values, objects or
allocations. Call sites read upstream slots through edges; their formal slot
lists never claim ownership of another definition's public interface.
"""
from copy import deepcopy
import json

from scar.ir.record_codec import encode
from scar.ir.v2.common import EvidenceClaim, EvidenceKind
from scar.ir.v2.contracts import EffectSummary
from scar.ir.v2.ids import OperationDefinitionID, ValueSlotID
from scar.ir.v2.semantic import (SemanticGraph, OperationDefinition, OperationKind,
    ValueSlot, SemanticEdge, SemanticEndpoint, SemanticNodeKind as N, SemanticRelation as R)
from scar.optimization.resource_solver import semantic_sha256


def project_source_pipeline_graph(pipeline, region_document, source_values):
    from scar.analysis.pipeline_region_pass import read_fusion_region, _contract
    from scar.compiler.pipeline_model import pipeline_digest

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

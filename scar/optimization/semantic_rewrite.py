"""Apply a proven closed static region replacement to existing v2 entities."""
from copy import deepcopy
from dataclasses import replace
import heapq

from scar.ir.execution_recipe import read_artifact
from scar.ir.v2.codec import semantic_from_dict
from scar.ir.v2.common import ProofStatus
from scar.ir.v2.optimization import TransformKind
from scar.ir.v2.semantic import (SemanticEdge, SemanticEndpoint, SemanticNodeKind as N,
    SemanticRelation as R)
from scar.optimization.resource_solver import semantic_sha256


def rewrite_closed_region(graph, root, replacement, alternative, *, original_graph, body_expansion):
    """Replace a complete flat static body, preserving its public interface.

    The selection and source producer own legality and profitability. This
    adapter rechecks structural applicability; it does not authorize execution.
    """
    before=deepcopy(graph.to_dict())
    if read_artifact(original_graph)!=before:
        raise ValueError('expansion/fallback must reference this exact original semantic graph')
    delta=alternative.delta
    unsupported=(delta.removed_instances,delta.substitutions,delta.moves,
        delta.static_substitutions,delta.static_moves)
    if (alternative.transform is not TransformKind.FUSE or any(unsupported)
            or not alternative.proofs or any(row.status is not ProofStatus.PROVEN for row in alternative.proofs)
            or not alternative.instructions or alternative.validation_request is None):
        raise ValueError('supported proven static complete-region fusion required')
    parent=graph.definitions[root]
    removed=set(delta.removed_definitions)
    if (removed!={node.id for node in graph.children_of(root)} or not removed
            or tuple(delta.added_operations)!=(replacement.id.value,)
            or replacement.id in graph.definitions or replacement.parent_id!=root
            or replacement.contract!=parent.contract or replacement.effect_summary!=parent.effect_summary
            or replacement.input_slots or replacement.output_slots or replacement.state_slots):
        raise ValueError('replacement must cover the complete flat source body and retain its conditional Q/effects')
    if any(graph.children_of(node) for node in removed):
        raise ValueError('nested member bodies require an explicit recursive replacement map')
    public=set(parent.input_slots+parent.output_slots+parent.state_slots)
    private={identity for identity,slot in graph.slots.items() if slot.owner in removed}
    if public.intersection(private):raise ValueError('private deletion overlaps public interface')
    if (any(control.owner in removed for control in graph.controls.values())
            or any(module.initializer in removed for module in graph.modules.values())
            or any(private.intersection(contract.required_outputs) for contract in graph.contracts.values())):
        raise ValueError('unmapped control, module initializer or required output refers to removed source members')
    for node in graph.definitions.values():
        if node.id not in removed and private.intersection(node.input_slots+node.output_slots+node.state_slots):
            raise ValueError('private source value escapes into another definition')
    private_refs={reference for key in private for reference in (key.wire,key.value)}
    for summary in graph.effects.values():
        for field in ('reads','writes','allocates','frees','aliases','escapes'):
            if any(target.kind.value=='value_slot' and target.reference in private_refs
                    for target in getattr(summary,field).members):
                raise ValueError('effect target needs an explicit private-slot replacement map')
    touches=lambda endpoint: endpoint.id in (removed if endpoint.kind is N.OPERATION else private if endpoint.kind is N.VALUE_SLOT else ())
    boundary={};deleted=[]
    for edge in graph.edges.values():
        if not (touches(edge.source) or touches(edge.target)):continue
        deleted.append(edge.edge_id)
        if edge.source.kind is N.OPERATION and edge.source.id in removed:
            if edge.target.kind is N.VALUE_SLOT and edge.target.id in public:
                if edge.relation not in (R.CONSUMES,R.PRODUCES,R.READS_SLOT,R.WRITES_SLOT):
                    raise ValueError('public source boundary has an unmapped relation')
                boundary.setdefault((edge.relation,edge.target),[]).append(edge)
            elif edge.target.kind is N.VALUE_SLOT and edge.target.id in private:pass
            elif edge.target.kind is N.OPERATION and edge.target.id in removed:pass
            elif edge.relation is R.HAS_CONTRACT and edge.target.id==parent.contract:pass
            elif edge.relation is R.HAS_EFFECT and edge.target.id==parent.effect_summary:pass
            else:raise ValueError('removed source operation has an unmapped external dependency')
        elif edge.source.kind is N.OPERATION and edge.source.id==root and edge.relation is R.CONTAINS:
            pass
        else:raise ValueError('removed source member/slot has an unhandled incoming edge or escape')
    result=semantic_from_dict(deepcopy(before))
    for key in deleted:del result.edges[key]
    for key in private:del result.slots[key]
    for key in removed:del result.definitions[key]
    metadata=deepcopy(replacement.metadata)
    metadata.update(source_expansion=dict(graph=original_graph,members=sorted(node.wire for node in removed)),
        lowered_body_expansion=body_expansion,
        retained_guards=[row.as_dict() for row in alternative.guards],
        retained_residual_effects=[row.as_dict() for row in alternative.residual_effects],
        runtime_guards_discharged=False,execution_authorized=False)
    result.add_definition(replace(replacement,metadata=metadata))
    first=next(iter(graph.edges.values())).evidence
    def add(relation,target,kind,role,evidence=first):
        identity='rewrite-edge:'+semantic_sha256(dict(operation=replacement.id.wire,
            relation=relation.value,target=target.wire,role=role))
        result.add_edge(SemanticEdge(identity,relation,SemanticEndpoint(N.OPERATION,replacement.id),
            SemanticEndpoint(kind,target),evidence))
        return identity
    edge_map={}
    for (relation,target),old in sorted(boundary.items(),key=lambda pair:(pair[0][0].value,pair[0][1].id.wire)):
        identity=add(relation,target.id,N.VALUE_SLOT,'public-boundary',old[0].evidence)
        edge_map.update({edge.edge_id:identity for edge in old})
    for relation,target,kind in ((R.HAS_CONTRACT,parent.contract,N.CONTRACT),
                               (R.HAS_EFFECT,parent.effect_summary,N.EFFECT)):
        if target is not None:add(relation,target,kind,'retained-Q')
    result.add_edge(SemanticEdge('rewrite-contains:'+replacement.id.wire,R.CONTAINS,
        SemanticEndpoint(N.OPERATION,root),SemanticEndpoint(N.OPERATION,replacement.id),first))
    result.assert_valid()
    if (any(result.slots[key].as_dict()!=graph.slots[key].as_dict() for key in public)
            or result.definitions[root].as_dict()!=parent.as_dict()):
        raise ValueError('public slots or region contract changed during rewrite')
    mapping=dict(removed_definitions=sorted(key.wire for key in removed),
        added_definition=replacement.id.as_dict(),removed_private_slots=sorted(key.wire for key in private),
        preserved_public_slots=sorted(key.wire for key in public),removed_edges=sorted(deleted),
        replaced_public_edges=edge_map,source_graph_unchanged=graph.to_dict()==before,
        original_graph=original_graph,body_expansion=body_expansion,
        runtime_guard_assumptions_retained=True,execution_authorized=False)
    return result,mapping


def analyze_static_region(graph, root):
    """Recompute source use/def and last-use positions for this exact revision."""
    graph.assert_valid();nodes={node.id for node in graph.children_of(root)}
    predecessors={key:set() for key in nodes}
    owners=nodes|{root}
    data={slot.slot_id:slot for slot in graph.slots.values() if slot.metadata.get('axis')=='data'
        and slot.owner in owners}
    definitions={key:set() for key in data};uses={key:set() for key in data}
    for edge in graph.edges.values():
        if (edge.source.kind is N.OPERATION and edge.source.id in nodes
                and edge.target.kind is N.VALUE_SLOT and edge.target.id in data):
            if edge.relation is R.PRODUCES:definitions[edge.target.id].add(edge.source.id)
            elif edge.relation in (R.CONSUMES,R.READS_SLOT):uses[edge.target.id].add(edge.source.id)
    facts=[]
    for identity,slot in data.items():
        producers=definitions[identity];consumers=uses[identity]
        if len(producers)>1:raise ValueError('source data slot has multiple unversioned producers')
        for consumer in consumers:predecessors[consumer].update(producers-{consumer})
        facts.append((slot,producers,consumers))
    for edge in graph.edges.values():
        if edge.relation is R.CONTROLS and edge.source.id in nodes and edge.target.id in nodes:
            predecessors[edge.target.id].add(edge.source.id)
    successors={key:set() for key in nodes};indegree={key:len(value) for key,value in predecessors.items()}
    for node,parents in predecessors.items():
        for parent in parents:successors[parent].add(node)
    by_wire={key.wire:key for key in nodes};ready=[key.wire for key in nodes if not indegree[key]]
    heapq.heapify(ready);order=[]
    while ready:
        key=by_wire[heapq.heappop(ready)];order.append(key)
        for child in successors[key]:
            indegree[child]-=1
            if not indegree[child]:heapq.heappush(ready,child.wire)
    if len(order)!=len(nodes):
        raise ValueError('static region data/source-order cycle requires a structured loop model')
    positions={key:index for index,key in enumerate(order)};public_outputs=set(graph.definitions[root].output_slots)
    rows=[dict(slot=slot.slot_id.as_dict(),producers=sorted(key.wire for key in producers),
        consumers=sorted(key.wire for key in consumers),
        last_use='REGION_EXIT' if slot.slot_id in public_outputs else
            max((positions[key] for key in consumers),default=None),
        no_consumer_does_not_prove_dead=True) for slot,producers,consumers in facts]
    return dict(schema='scar.semantic-region-analysis',schema_version=1,
        revision=semantic_sha256(graph.to_dict()),root=root.as_dict(),
        operation_order=[key.wire for key in order],data_use_def=rows,
        world_state_aliases_inferred=False,opaque_effects_preserved=True,
        timing_or_storage_lifetime_inferred=False)


def require_current_analysis(graph, analysis, *, root):
    if (analysis['revision']!=semantic_sha256(graph.to_dict())
            or analysis['root']!=root.as_dict()):
        raise ValueError('analysis belongs to a different graph revision or region')
    return analysis

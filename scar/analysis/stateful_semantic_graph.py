"""Source-bound stateful regions projected into the existing v2 SG.

Evaluation actions and their arithmetic expressions are the same operation.
Descriptors and shared-call expansion references are static definitions, not
additional executions. Array epochs belong to this region template only.
Original F4 admission remains the enclosing controller's responsibility.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import json

from scar.compiler.stateful_model import validate_stateful_region, stateful_digest
from scar.ir.execution_recipe import read_artifact
from scar.ir.v2.codec import semantic_from_dict
from scar.ir.v2.common import EvidenceClaim, EvidenceKind
from scar.ir.v2.contracts import (ContractDefinition, ContractFacet, FacetRequirement, EffectSummary)
from scar.ir.v2.ids import OperationDefinitionID, ValueSlotID, ContractID, EffectSummaryID
from scar.ir.v2.semantic import (SemanticGraph, OperationDefinition, OperationKind, ValueSlot,
    SemanticEdge, SemanticEndpoint, SemanticNodeKind as N, SemanticRelation as R)
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.stateful_schedule import _metadata, _normalized


SCHEMA = 'scar.stateful-semantic-graph'
_RULE = 'EXPLICIT_ORIGINAL_STATEFUL_F4_TO_STATIC_V2_ENTITIES_V1'
_DESCRIPTORS = frozenset(('input', 'const', 'slice'))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _wire(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _source(source):
    return {key:deepcopy(value) for key, value in asdict(source).items() if key != 'origins'}


def _origin_key(origin):
    return semantic_sha256({key:origin.get(key) for key in ('unit', 'template', 'region', 'step', 'source')})


def _caller_ports(region):
    """Select original caller ports structurally, never from action adjacency."""
    identities = [row for row in region.source.origins if row.get('kind') == 'source_identity'
        and row.get('code_key') == region.source.code_key]
    entry, returns = {}, {}
    for identity in identities:
        for index, origin in enumerate(region.source.origins):
            if (origin.get('kind') != 'original_f4' or origin.get('unit') != identity.get('unit')
                    or origin.get('template', {}).get('occurrence') != identity.get('occurrence')):
                continue
            before = origin.get('world_before')
            if (before is not None and before.get('port', {}).get('kind') == 'PARAMETER'
                    and before['port'].get('ordinal') == len(region.parameter_names) + 1):
                entry.setdefault(semantic_sha256(before), (before, index))
            results = [value for value in origin.get('results', ())
                if value.get('port', {}).get('completion') == 'RETURN']
            if len(results) == 2 and [value['port']['ordinal'] for value in results] == [0, 1]:
                returns.setdefault(semantic_sha256(results), (results, index))
    return (next(iter(entry.values())) if len(entry) == 1 else None,
        next(iter(returns.values())) if len(returns) == 1 else None)


def derive_stateful_semantic_graph(region, *, input_shapes, source_refs, source_proof=None):
    """Derive metadata only; source_proof is a retained parent check receipt.

    A supplied observed-specialization check is matched by region and shapes.
    Its original F4 cold replay is not repeated or claimed by this adapter.
    Missing shapes/origins/world boundaries stay explicit frontier entries.
    """
    region = validate_stateful_region(region)
    _need(type(source_refs) in (tuple, list) and source_refs
        and all(type(row) is dict and row for row in source_refs),
        'original source/proof artifact references are required')
    source_refs = _wire(source_refs)
    region_sha = stateful_digest(region)
    material = dict(schema='scar.stateful-array-region', semantic_sha256=region_sha)
    for row in source_refs:
        reference = row.get('artifact', row)
        if reference.get('schema') == 'scar.stateful-array-region' and 'uri' in reference:
            from scar.compiler.stateful_model import stateful_from_dict
            _need(stateful_digest(stateful_from_dict(read_artifact(reference))) == region_sha,
                'original region artifact differs from the projected source')
            material = dict(artifact=deepcopy(reference))
            break
    inputs, nodes, declared, shapes, kinds, _, _, _, shape_issues = _metadata(region, input_shapes)
    frontier = [dict(field='shape.' + row['field'], reason=row['reason']) for row in shape_issues]
    if source_proof is not None:
        _need(type(source_proof) is dict and source_proof.get('region_id') == region.id
            and source_proof.get('input_shapes') == declared,
            'parent source/shape check belongs to a different region specialization')
    graph = SemanticGraph()
    root = OperationDefinitionID('stateful-source-region:' + region_sha)
    cid, eid = ContractID('stateful-Q:' + region_sha), EffectSummaryID('stateful-unknown-effects:' + region_sha)
    evidence = EvidenceClaim(EvidenceKind.INFERRED,
        references=tuple(json.dumps(row, sort_keys=True, separators=(',', ':')) for row in source_refs),
        scope=root.wire, assumptions=tuple(region.guards))

    def ref(pointer):
        return {**deepcopy(material), 'locator':pointer}

    def identity(cls, role, original):
        return cls('stateful-source-projection:' + semantic_sha256(dict(
            rule=_RULE, region_sha256=region_sha, role=role, original=original)))

    def edge(relation, source, target, role, source_kind=N.OPERATION, target_kind=N.VALUE_SLOT):
        key = 'stateful-source-edge:' + semantic_sha256(dict(relation=relation.value,
            source=source.wire, target=target.wire, role=role))
        graph.add_edge(SemanticEdge(key, relation, SemanticEndpoint(source_kind, source),
            SemanticEndpoint(target_kind, target), evidence))
        return key

    def operation(key, kind, label, metadata, *, source=None, parent=root):
        graph.add_definition(OperationDefinition(key, kind, label,
            code_id=None if source is None else source.code_key,
            source_file=region.source.path if source is None else source.path,
            parent_id=parent, contract=cid, effect_summary=eid, metadata=metadata))
        if parent is not None:
            edge(R.CONTAINS, parent, key, 'static-containment', target_kind=N.OPERATION)
        edge(R.HAS_CONTRACT, key, cid, 'conditional-source-Q', target_kind=N.CONTRACT)
        edge(R.HAS_EFFECT, key, eid, 'effects-remain-unknown', target_kind=N.EFFECT)

    def slot(role, original, owner, kind, metadata):
        key = identity(ValueSlotID, role, original)
        graph.add_slot(ValueSlot(key, role, role, semantic_type=kind, owner=owner,
            metadata=dict(runtime_logical_version=False, physical_storage='UNKNOWN',
                physical_position='UNKNOWN', **metadata)))
        edge(R.CONTAINS, owner, key, 'slot-owner')
        return key

    graph.add_effect(EffectSummary(eid, evidence=(evidence,)))
    graph.add_contract(ContractDefinition(cid, 'Original guarded stateful region Q',
        facets=tuple(FacetRequirement(facet, True, description=description) for facet, description in (
            (ContractFacet.OUTPUT, 'Normal return None and the original final array state.'),
            (ContractFacet.STATE, 'Original ordered writes, alias descriptors and region-local epochs.'),
            (ContractFacet.NUMERICAL, 'Original operands/providers and strict guarded floating-point/fallback contract; no reassociation.'),
            (ContractFacet.EXCEPTION, 'Original guarded normal-completion domain; exceptional paths remain outside this projection.'),
            (ContractFacet.ORDERING, 'Original source action order is retained.'),
            (ContractFacet.EXTERNAL, 'Unknown whole-program/provider/world effects remain explicit.'))),
        assumptions=tuple(region.guards), evidence=(evidence,)))
    operation(root, OperationKind.REGION, 'Source-qualified stateful region',
        dict(source=_source(region.source), source_refs=source_refs, source_region=ref('#'),
            source_admission='PARENT_ORIGINAL_F4_COLD_ADMISSION_REQUIRED',
            runtime_guards=list(region.guards), runtime_guards_discharged=False,
            source_bodies_copied=False, scope='ONE_ORIGINAL_STATEFUL_REGION_TEMPLATE_NORMAL_COMPLETION'),
        source=region.source, parent=None)
    public_inputs, epochs, input_rows = {}, {}, []
    for index, row in enumerate(region.inputs):
        metadata = dict(input_id=row.id, input_path=list(row.path), value_kind=row.kind,
            source_refs=[ref('#/inputs/' + str(index))],
            scalar_value_specialized=False, value='UNKNOWN', loaded_each_call=True,
            shape=declared.get(row.id) if row.kind == 'array' else None,
            shape_qualification='CONDITIONAL_INPUT_SHAPE_NOT_RUNTIME_INVARIANCE',
            dtype='float64' if row.kind == 'array' else 'GUARDED_FINITE_PYTHON_INT_OR_FLOAT',
            region_local_epoch=0 if row.kind == 'array' else None)
        key = slot('input', dict(index=index, path=list(row.path)), root,
            'region-local-array-state' if row.kind == 'array' else 'per-call-scalar-input', metadata)
        public_inputs[row.id] = key
        if row.kind == 'array':epochs[row.id, 0] = key
        input_rows.append(dict(input=row.id, slot=key.as_dict(), source_refs=metadata['source_refs']))
    for index, action in enumerate(region.actions):
        if action.kind == 'write':
            epochs[action.target, action.epoch_after] = slot('state-epoch',
                dict(input=action.target, epoch=action.epoch_after), root, 'region-local-array-state',
                dict(input_id=action.target, input_path=list(inputs[action.target].path),
                    region_local_epoch=action.epoch_after, global_content_identity=False,
                    fresh_allocation_inferred=False, shape=declared.get(action.target),
                    source_refs=[ref('#/actions/' + str(index))]))
    origins = defaultdict(list)
    for index, origin in enumerate(region.source.origins):
        if origin.get('expression') in nodes:origins[origin['expression']].append(index)
    evaluations = {action.value:(index, action) for index, action in enumerate(region.actions) if action.kind == 'evaluate'}
    expression_ids, expression_slots, expression_rows = {}, {}, []
    for index, node in enumerate(region.nodes):
        key = identity(OperationDefinitionID, 'expression', dict(id=node.id, source_pointer='#/nodes/' + str(index)))
        expression_ids[node.id] = key
        origin_refs = [ref('#/source/origins/' + str(i)) for i in origins[node.id]]
        if not origin_refs:
            frontier.append(dict(field='expressions.' + node.id, reason='ORIGINAL_F4_EXPRESSION_ORIGIN_MISSING'))
        action_entry = evaluations.get(node.id)
        metadata = dict(original_expression=node.id, opcode=node.op, attributes=deepcopy(node.attrs),
            ordered_operands=list(node.inputs), value_kind=kinds[node.id], shape=shapes[node.id],
            source_refs=[ref('#/nodes/' + str(index)), *origin_refs],
            execution_member=action_entry is not None,
            descriptor_definition=node.op in _DESCRIPTORS,
            scalar_input_is_constant=False if node.op == 'input' and kinds[node.id] == 'scalar' else None,
            additional_evaluation_operation_created=False)
        if action_entry is not None:
            metadata.update(original_evaluate_action=action_entry[1].id,
                evaluate_action_source_ref=ref('#/actions/' + str(action_entry[0])),
                expression_and_evaluate_action_are_one_operation=True)
        if node.op == 'slice':
            base = nodes[node.inputs[0]]
            while base.op == 'slice':base = nodes[base.inputs[0]]
            metadata['alias_descriptor'] = dict(base_input=base.attrs['name'],
                region_local_epoch=base.attrs['epoch'], bounds=deepcopy(node.attrs['bounds']),
                parent_descriptor=node.inputs[0], reads_at_original_consumption_epoch=True,
                fresh_array=False, object_identity_or_address_inferred=False)
        kind = OperationKind.CONSTANT if node.op == 'const' else OperationKind.INDEX if node.op == 'slice' else OperationKind.OPERATOR
        operation(key, kind, 'Static ' + node.op, metadata)
        result = slot('expression-result', node.id, key,
            'source-alias-descriptor' if node.op in ('input', 'slice') and kinds[node.id] == 'array' else 'source-expression-value',
            dict(expression=node.id, value_kind=kinds[node.id], shape=shapes[node.id],
                immutable_computed_value=node.op not in _DESCRIPTORS,
                allocation_identity_inferred=False, source_refs=[ref('#/nodes/' + str(index))]))
        expression_slots[node.id] = result
        graph.definitions[key].output_slots = (result,)
        edge(R.PRODUCES, key, result, 'expression-result')
        for ordinal, operand in enumerate(node.inputs):
            edge(R.CONSUMES, key, expression_slots[operand], 'operand:' + str(ordinal))
        if node.op == 'input':
            source = epochs[node.attrs['name'], node.attrs['epoch']] if kinds[node.id] == 'array' else public_inputs[node.attrs['name']]
            edge(R.READS_SLOT, key, source, 'original-input-path-and-local-epoch')
        expression_rows.append(dict(expression=node.id, operation=key.as_dict(), result=result.as_dict(),
            evaluate_action=None if action_entry is None else action_entry[1].id,
            origin_refs=origin_refs, source_refs=[ref('#/nodes/' + str(index))]))
    action_rows, execution_members, world_rows, previous = [], [], [], None
    for index, action in enumerate(region.actions):
        if action.kind == 'evaluate':
            key = expression_ids[action.value]
        else:
            key = identity(OperationDefinitionID, 'ordered-write', dict(id=action.id, source_pointer='#/actions/' + str(index)))
            shape = declared.get(action.target)
            normalized, target_shape = _normalized(action.bounds, shape) if shape is not None else (None, None)
            if shape is not None and kinds[action.value] == 'array' and shapes[action.value] != target_shape:
                frontier.append(dict(field='actions.' + action.id + '.shape', reason='WRITE_RHS_AND_ORIGINAL_TARGET_SHAPES_DIFFER'))
            operation(key, OperationKind.OPERATOR, 'Original ordered array write',
                dict(original_action=action.id, write_operator=action.op, input_id=action.target,
                    input_path=list(inputs[action.target].path), bounds=deepcopy(action.bounds),
                    normalized_bounds=normalized, target_shape=target_shape,
                    epoch_before=action.epoch_before, epoch_after=action.epoch_after,
                    epoch_scope='THIS_REGION_TEMPLATE_ONLY_NOT_GLOBAL_LOGICAL_VERSION',
                    source_refs=[ref('#/actions/' + str(index))], execution_member=True,
                    fresh_allocation_inferred=False))
            edge(R.CONSUMES, key, expression_slots[action.value], 'write-rhs')
            edge(R.READS_SLOT, key, epochs[action.target, action.epoch_before], 'original-target-before-write')
            edge(R.WRITES_SLOT, key, epochs[action.target, action.epoch_after], 'original-target-after-write')
            edge(R.PRODUCES, key, epochs[action.target, action.epoch_after], 'region-local-state-definition')
        if previous is not None:edge(R.CONTROLS, previous, key, 'ORIGINAL_SOURCE_ACTION_ORDER', target_kind=N.OPERATION)
        previous = key
        action_world_slots = []
        for side, endpoint in (('before', action.world_before), ('after', action.world_after)):
            if endpoint is None:
                frontier.append(dict(field='actions.' + action.id + '.world_' + side,
                    reason='ORIGINAL_WORLD_ENDPOINT_NOT_RETAINED_NOT_INFERRED_FROM_ADJACENCY'))
                continue
            world = slot('world-endpoint', dict(action=action.id, side=side, original=endpoint), key,
                'original-f4-world-endpoint', dict(original_port=deepcopy(endpoint),
                    source_refs=[ref('#/actions/' + str(index) + '/world_' + side)],
                    no_adjacent_world_equality_inferred=True))
            edge(R.READS_SLOT if side == 'before' else R.WRITES_SLOT, key, world, 'original-world-' + side)
            action_world_slots.append(world)
            world_rows.append(dict(action=action.id, side=side, slot=world.as_dict(), original_port=deepcopy(endpoint)))
        graph.definitions[key].state_slots = tuple(action_world_slots)
        execution_members.append(key)
        action_rows.append(dict(action=action.id, kind=action.kind, operation=key.as_dict(),
            expression=action.value, source_refs=[ref('#/actions/' + str(index))],
            shares_expression_operation=action.kind == 'evaluate'))
    callee_rows, provider_rows, shared = [], [], {}
    callsite_index = {}
    for index, callee in enumerate(region.callees):
        source_identity = _source(callee.source)
        share = semantic_sha256(source_identity)
        if share not in shared:
            shared[share] = identity(OperationDefinitionID, 'shared-callee-reference', source_identity)
            operation(shared[share], OperationKind.FUNCTION, 'Shared original source callee',
                dict(source=source_identity, source_refs=[ref('#/callees/' + str(index) + '/source')],
                    original_body_reference_only=True, static_body_copied=False,
                    runtime_invocation_inferred=False, execution_member=False), source=callee.source, parent=None)
        sites = []
        for oi, origin in enumerate(callee.origins):
            site = identity(OperationDefinitionID, 'source-callsite-reference', _origin_key(origin))
            if site not in graph.definitions:
                operation(site, OperationKind.REGION, 'Original shared-callee expansion reference',
                    dict(source_refs=[ref('#/callees/' + str(index) + '/origins/' + str(oi))],
                        execution_member=False, expansion_already_represented_by_region_expressions=True,
                        additional_call_work_inferred=False, runtime_invocation_inferred=False))
                edge(R.CALLS, site, shared[share], 'original-shared-callee-reference', target_kind=N.OPERATION)
            callsite_index[_origin_key(origin)] = site
            sites.append(site.as_dict())
        callee_rows.append(dict(callee_index=index, lookup_path=list(callee.path),
            shared_definition=shared[share].as_dict(), callsite_references=sites,
            source_refs=[ref('#/callees/' + str(index))]))
    for node in region.nodes:
        references = []
        for oi in origins[node.id]:
            origin = region.source.origins[oi]
            if origin.get('callsite'):
                site = callsite_index.get(_origin_key(origin['callsite']))
                if site is None:
                    frontier.append(dict(field='expressions.' + node.id + '.callee', reason='SHARED_CALLEE_CALLSITE_REFERENCE_UNRESOLVED'))
                else:references.append(site.as_dict())
        if references:graph.definitions[expression_ids[node.id]].metadata['existing_source_expansion_references'] = references
    for index, provider in enumerate(region.providers):
        key = identity(OperationDefinitionID, 'shared-provider-reference', dict(
            id=provider.id, module=provider.module, symbol=provider.symbol, semantics=provider.semantics))
        operation(key, OperationKind.OPAQUE, 'Original guarded provider reference',
            dict(provider_id=provider.id, lookup_path=list(provider.lookup_path), semantics=provider.semantics,
                source_refs=[ref('#/providers/' + str(index))], static_body_copied=False,
                execution_member=False, provider_effects='UNKNOWN'), parent=None)
        for node in region.nodes:
            if node.op == 'power2' and node.attrs['provider'] == provider.id:
                edge(R.CALLS, expression_ids[node.id], key, 'original-provider-call', target_kind=N.OPERATION)
        provider_rows.append(dict(provider=provider.id, definition=key.as_dict(),
            source_refs=[ref('#/providers/' + str(index))]))
    entry, returned = _caller_ports(region)
    caller_world = {}
    for side, item in (('entry', entry), ('exit', returned)):
        endpoint = None if item is None else item[0] if side == 'entry' else item[0][1]
        if endpoint is None:
            frontier.append(dict(field='caller.world_' + side, reason='EXACT_ORIGINAL_CALLER_WORLD_BOUNDARY_NOT_UNIQUELY_RETAINED'))
        key = slot('caller-world', side, root, 'original-f4-caller-world-port',
            dict(original_port=deepcopy(endpoint), status='UNKNOWN' if endpoint is None else 'SOURCE_PORT_RETAINED',
                source_refs=[] if item is None else [ref('#/source/origins/' + str(item[1]))],
                no_action_endpoint_substitution=True))
        caller_world[side] = key
        edge(R.READS_SLOT if side == 'entry' else R.WRITES_SLOT, root, key, 'original-caller-world-' + side)
    return_slot = slot('normal-return', 'NONE', root, 'original-normal-return-none',
        dict(return_kind=region.return_kind, original_port=None if returned is None else deepcopy(returned[0][0]),
            source_refs=[ref('#/return_kind')] + ([] if returned is None else [ref('#/source/origins/' + str(returned[1]))])))
    return_op = identity(OperationDefinitionID, 'normal-return', region.return_kind)
    operation(return_op, OperationKind.CONSTANT, 'Original normal None return',
        dict(literal=None, source_refs=[ref('#/return_kind')], execution_member=False,
            return_control_anchor_only=True, additional_arithmetic_work=False))
    edge(R.PRODUCES, return_op, return_slot, 'normal-return-none')
    if previous is not None:edge(R.CONTROLS, previous, return_op, 'normal-return-after-original-actions', target_kind=N.OPERATION)
    written = {action.target:action.epoch_after for action in region.actions if action.kind == 'write'}
    final_state = [epochs[name, written[name]] for name in sorted(written)]
    graph.definitions[root].input_slots = tuple(public_inputs.values())
    graph.definitions[root].output_slots = (return_slot,)
    graph.definitions[root].state_slots = (*final_state, caller_world['entry'], caller_world['exit'])
    graph.contracts[cid].required_outputs = (return_slot, *final_state)
    graph.definitions[root].metadata.update(execution_members=[key.as_dict() for key in execution_members],
        scalar_inputs_loaded_each_call=[row.id for row in region.inputs if row.kind == 'scalar'],
        epoch_scope='THIS_REGION_TEMPLATE_ORIGINAL_ORDERED_ROOT_WRITES_ONLY_NOT_GLOBAL_LOGICAL_VERSION')
    graph.assert_valid()
    coverage = dict(inputs=dict(total=len(region.inputs), projected=len(input_rows)),
        expressions=dict(total=len(region.nodes), projected=len(expression_rows),
            with_original_F4_origins=sum(bool(row['origin_refs']) for row in expression_rows)),
        actions=dict(total=len(region.actions), projected=len(action_rows),
            evaluations_share_expression_operation=sum(row['shares_expression_operation'] for row in action_rows)),
        callees=dict(total=len(region.callees), projected=len(callee_rows), shared_definitions=len(shared)),
        providers=dict(total=len(region.providers), projected=len(provider_rows)),
        execution_operations=len(execution_members),
        source_descriptor_definitions=sum(node.op in _DESCRIPTORS for node in region.nodes),
        original_whole_program_gaps_reassessed=False, whole_program_coverage_complete=False)
    for field, reason in (
        ('whole_program', 'Only this admitted closed region is projected; original whole-program gaps are not removed or reassessed.'),
        ('physical_storage', 'Region-local slots/epochs do not establish allocation identity, address, residency, capacity or peak RSS.'),
        ('hardware_demand', 'Static operations do not establish whole-entry CPU demand, thread placement or physical duration.'),
        ('world_effects', 'Retained source world endpoints and order do not close unknown external/provider/world effects.')):
        frontier.append(dict(field=field, reason=reason))
    result = dict(schema=SCHEMA, schema_version=1, rule=_RULE, region_id=region.id,
        region_sha256=region_sha, root=root.as_dict(), graph=graph.to_dict(),
        graph_revision=semantic_sha256(graph.to_dict()),
        source_refs=source_refs, source_proof=deepcopy(source_proof),
        input_shapes=declared, shape_qualification='CONDITIONAL_ON_RETAINED_RUNTIME_GUARDS_NOT_RUNTIME_INVARIANCE',
        projection=dict(inputs=input_rows, expressions=expression_rows, actions=action_rows,
            callees=callee_rows, providers=provider_rows,
            execution_members=[key.as_dict() for key in execution_members],
            state_epochs=[dict(input=name, epoch=epoch, slot=key.as_dict()) for (name, epoch), key in sorted(epochs.items())],
            action_world_endpoints=world_rows, caller_world_slots={key:value.as_dict() for key, value in caller_world.items()},
            normal_return=return_slot.as_dict()),
        qualification=dict(source_admission='PARENT_COLD_ADMISSION_REQUIRED_NOT_PERFORMED_HERE',
            parent_source_check_supplied=source_proof is not None, runtime_guards=list(region.guards),
            runtime_guards_discharged=False, effects='UNKNOWN', normal_completion_only=True,
            whole_program_effects_closed=False, scalar_values_specialized=False,
            source_bodies_copied=False, imported_callee_bodies_reinlined=False,
            runtime_invocations_created=False, runtime_logical_versions_created=False,
            physical_storage_or_position_inferred=False),
        coverage=coverage, frontier=frontier, target_executed=False, native_loaded=False,
        compiled=False, execution_authorized=False, deployment_authorized=False)
    return _wire(result)


def read_stateful_semantic_graph(document):
    _need(type(document) is dict and (document.get('schema'), document.get('schema_version')) == (SCHEMA, 1),
        'versioned stateful semantic graph projection required')
    graph = semantic_from_dict(document['graph'])
    _need(semantic_sha256(graph.to_dict()) == document['graph_revision'],
        'stateful graph revision differs from its actual SemanticGraph')
    root = OperationDefinitionID(document['root']['value'])
    _need(root in graph.definitions and all(OperationDefinitionID(row['value']) in graph.definitions
        for row in document['projection']['execution_members']), 'root or source execution member is missing')
    return graph


def verify_stateful_semantic_graph(region, document, *, input_shapes, source_refs, source_proof=None):
    expected = derive_stateful_semantic_graph(region, input_shapes=input_shapes,
        source_refs=source_refs, source_proof=source_proof)
    _need(semantic_sha256(expected) == semantic_sha256(document),
        'stateful semantic graph differs from original expressions/actions/ports/shape replay')
    read_stateful_semantic_graph(document)
    return expected


__all__ = ['SCHEMA', 'derive_stateful_semantic_graph', 'read_stateful_semantic_graph',
    'verify_stateful_semantic_graph']
